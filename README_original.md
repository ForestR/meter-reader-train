项目重点：本仓库聚焦于YOLO模型训练流程中的下游任务微调阶段。

具体说明：我们采用迁移学习，加载了在通用数据集上预训练好的YOLO官方权重(e.g., YOLO11, YOLO26)作为模型起点。

---

## 模型架构

We have 2 types of final model, end2end model and pipeline model. 

- The end2end model is a single model that takes the snapshot as input and outputs the meter reading. 
- The pipeline model is a three-stage model that first detects the meter reading from the snapshot, then split to digit segments, and finally recognizes the meter reading from sorted digit segments.

```
# Model Configuration
# Model type: 'end2end' for single-stage YOLO11 model, 'pipeline' for three-stage pipeline
MODEL_TYPE=pipeline
# Path to model weights - use end2end model for single-stage, or pipeline directory for three-stage
MODEL_PATH_END2END=checkpoints/3k_dataset/end2end_yolo11l.pt
MODEL_PATH_PIPELINE=checkpoints/3k_dataset/pipeline_yolo11n/  # including {01_dial_seg.pt, 02_digit_seg.pt, 03_digit_cls.pt}
MODEL_DEVICE=auto  # cpu/cuda/auto
CONFIDENCE_THRESHOLD=0.8
MAX_BATCH_SIZE=32

# Image Processing
IMAGE_WIDTH=800
IMAGE_HEIGHT=600
IMAGE_MAX_SIZE_MB=10
SUPPORTED_FORMATS=JPEG,PNG,BMP
```

Well, although we call it a segmentation model (01_dial_seg.pt, 02_digit_seg.pt), it is actually a detection model (e.g., yolo26n.pt).  # comment: maybe we should use a more accurate term to name our checkpoints to avoid confusion?

We also have to build a script to automatically convert the standard dataset to a processed dataset for pipeline model training. # comment: we already have it.

---

## 潜在改进 1：训练策略

可以参考以下情况来做决策：

- 推荐冻结：如果抄表数据集规模较小（如数千张图片），或计算资源有限（如GPU内存不足）。此时，"冻结骨干网络，仅训练模型头部"能快速获得一个不错的基线模型。

- 推荐不冻结（完全微调）：如果抄表数据集规模较大，或任务场景与通用数据集（如COCO）差异极大（例如，电表型号独特、拍摄角度特殊）。此时，解冻训练能让模型更好地学习特定特征。

- 折中策略（推荐）：采用分阶段解冻的训练策略。先冻结骨干网络训练若干轮次，让模型头部快速适应；然后解冻骨干网络全部或部分层，用较小的学习率进行联合微调。这是许多实践中效果最佳的方法。

我们的选择：
最稳妥高效的方案是先冻结骨干网络进行训练。在冻结训练完成后，可以尝试解冻骨干网络，使用更小的学习率（如初始学习率的10%）进行第二轮微调。观察验证集精度是否有提升，以此判断完全微调对你的任务是否必要。

---

## 潜在改进 2：角度矫正

We used a batch of edge sensors (fixed positions) to collect a large number of images of the dial plates of gas flow meters. Most of the images have a certain degree (≤ ±15°) of horizontal skew. We hope to use the YOLO model to extract the readings of the mechanical dials in the display area of the dial plate.

问题在于：
我们应该选用标准的包围盒检测模型（yolo26n.pt），还是选用旋转包围盒检测模型（yolo26n-obb.pt）？

current:
我们目前使用的模型是标准的包围盒检测模型，且没有对输入图像进行任何的数字图像处理（e.g., 角度矫正）。

可能的改进方向：

1. 对输入图像进行角度矫正，然后使用标准包围盒检测模型进行训练。
2. 或者，使用旋转包围盒检测模型进行训练。

建议是：**优先选择“图像矫正 + 标准框模型”方案**。这通常比直接使用旋转框模型更简单、高效且更适合你的任务。

下面的对比分析可以帮你清晰理解这个建议：

| 特性           | **方案一：图像矫正 + 标准框模型**                            | **方案二：旋转包围盒 (OBB) 模型**                            |
| :------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| **核心思路**   | 前置处理，将倾斜图像**矫正**为标准水平状态，然后使用常规YOLO。 | 模型直接输出带角度的检测框，**适应**物体的原始倾斜。         |
| **标注成本**   | **低**。沿用现有的标准矩形框标注格式，无需改变。             | **高**。需要使用四点坐标或`xywhr`格式重新标注整个数据集。    |
| **模型复杂度** | **低**。使用成熟、轻量的标准检测模型，训练和推理速度快。     | **高**。OBB模型结构更复杂，计算量通常更大。                  |
| **任务匹配度** | **高**。矫正后，数字排列水平，与标准框假设完美匹配，符合你的OCR任务本质。 | **中/低**。**过度设计**：数字是离散目标，倾斜与否不影响其矩形边界，OBB的旋转优势无法发挥。 |
| **流程复杂度** | **简单**。增加一个自动化的预处理模块，训练和部署流程清晰。   | **复杂**。需引入新的标注、训练和推理流程，且后处理需处理角度。 |

---

## 潜在改进 3：数据集设计

聚焦于数据集的设计。

We hope to design the architecture of a repo used for model training based on YOLO. The model is trained to do OCR task to get meter reading from the snapshot collected by edge device. The standard dataset is in YOLO format:

```
data/
 ├─ dataset_1#/
 │   ├─ images/
 │   ├─ labels/
 │   ├─ README.md
 │   ├─ JSON or YAML
 ├─ dataset_2#/
 ├─ dataset_3#/
```

where,
``` data > xuzhou_3k > labels > value_025837_05.txt
0 0.251509 0.609492 0.045622 0.086208
2 0.333048 0.608764 0.043990 0.085955
5 0.411117 0.599102 0.046863 0.080664
8 0.486942 0.597155 0.044130 0.084428
3 0.561221 0.591790 0.041101 0.081476
7 0.635312 0.600504 0.045014 0.086745
```

注意：我们目前的数据集没有对输入图像进行任何的数字图像处理（e.g., 角度矫正）。即，BBox是标注于原始图像的。这在一定程度上可以提升模型本身的鲁棒性。

也许我们可用创建一个dedicated dirty dataset for negative samples? 当训练时，基于flag传入的argument，动态地混入clean dataset？(建议: 在标准YOLO格式基础上进行扩展，实现物理隔离与逻辑混合)

此外，也许我们也可以将这种数据集训练时动态混合的设计沿用到更垂直的任务场景中。譬如，我们在不同城市（常州、西安、徐州、苏州, etc.）、不同用户（燃气服务供应商）、不同设备（燃气表、水表、电表, etc.）、不同场景（室内、室外, etc.）下收集数据，然后根据下游任务混合多个数据集，有针对性的进行模型微调。

这种清晰隔离的数据集设计更利于扩展、维护和实验复现。

这涉及到项目生命周期的后半部分，即模型部署后的持续学习、动态更新与持续优化。我们会拿到大量的真实业务数据，这些数据往往存在各种问题（e.g., 角度倾斜、图像模糊、遮挡、噪声, etc.）。因此，我们需要设计一个灵活的数据集混合策略，以适应不同的任务需求和数据质量。(comment: 这点在pipeline模型中尤为重要，因为pipeline模型需要处理更复杂的场景，如：数字粘连、数字缺失、数字变形, etc.)

## 潜在改进 4：数据标注

既然我们需要持续的收集真实业务数据，那么我们是否需要考虑数据标注的自动化？

也许我们可以用经过微调的私有模型来标注数据，以此实现数据飞轮的闭环。

此外，我们还会遇到许多困难样本（即现有模型会犯错），他们往往是原始数据集难以覆盖的边缘case，对于模型的能力提升十分重要。

因此除了自动标注工具，我们还需要提供一个工具用于人工标注这些困难样本。

