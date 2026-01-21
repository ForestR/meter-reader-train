# Pose Estimation

![YOLO pose estimation with human body keypoint detection](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/pose-estimation-examples.avif)

Pose estimation is a task that involves identifying the location of specific points in an image, usually referred to as keypoints. The keypoints can represent various parts of the object such as joints, landmarks, or other distinctive features. The locations of the keypoints are usually represented as a set of 2D `[x, y]` or 3D `[x, y, visible]` coordinates.

The output of a pose estimation model is a set of points that represent the keypoints on an object in the image, usually along with the confidence scores for each point. Pose estimation is a good choice when you need to identify specific parts of an object in a scene, and their location in relation to each other.



**Watch:** Ultralytics YOLO26 Pose Estimation Tutorial | Real-Time Object Tracking and Human Pose Detection

Tip

YOLO26 *pose* models use the `-pose` suffix, i.e., `yolo26n-pose.pt`. These models are trained on the [COCO keypoints](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco-pose.yaml) dataset and are suitable for a variety of pose estimation tasks.

In the default YOLO26 pose model, there are 17 keypoints, each representing a different part of the human body. Here is the mapping of each index to its respective body joint:

1. Nose
2. Left Eye
3. Right Eye
4. Left Ear
5. Right Ear
6. Left Shoulder
7. Right Shoulder
8. Left Elbow
9. Right Elbow
10. Left Wrist
11. Right Wrist
12. Left Hip
13. Right Hip
14. Left Knee
15. Right Knee
16. Left Ankle
17. Right Ankle

## [Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/26)

Ultralytics YOLO26 pretrained Pose models are shown here. Detect, Segment and Pose models are pretrained on the [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) dataset, while Classify models are pretrained on the [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml) dataset.

[Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) download automatically from the latest Ultralytics [release](https://github.com/ultralytics/assets/releases) on first use.

| Model                                                        | size (pixels) | mAPpose 50-95(e2e) | mAPpose 50(e2e) | Speed CPU ONNX (ms) | Speed T4 TensorRT10 (ms) | params (M) | FLOPs (B) |
| :----------------------------------------------------------- | :------------ | :----------------- | :-------------- | :------------------ | :----------------------- | :--------- | :-------- |
| [YOLO26n-pose](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-pose.pt) | 640           | 57.2               | 83.3            | 40.3 ± 0.5          | 1.8 ± 0.0                | 2.9        | 7.5       |
| [YOLO26s-pose](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-pose.pt) | 640           | 63.0               | 86.6            | 85.3 ± 0.9          | 2.7 ± 0.0                | 10.4       | 23.9      |
| [YOLO26m-pose](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m-pose.pt) | 640           | 68.8               | 89.6            | 218.0 ± 1.5         | 5.0 ± 0.1                | 21.5       | 73.1      |
| [YOLO26l-pose](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l-pose.pt) | 640           | 70.4               | 90.5            | 275.4 ± 2.4         | 6.5 ± 0.1                | 25.9       | 91.3      |
| [YOLO26x-pose](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x-pose.pt) | 640           | 71.6               | 91.6            | 565.4 ± 3.0         | 12.2 ± 0.2               | 57.6       | 201.7     |

- **mAPval** values are for single-model single-scale on [COCO Keypoints val2017](https://cocodataset.org/) dataset.
  Reproduce by `yolo val pose data=coco-pose.yaml device=0`
- **Speed** averaged over COCO val images using an [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) instance.
  Reproduce by `yolo val pose data=coco-pose.yaml batch=1 device=0|cpu`

## Train

Train a YOLO26-pose model on the COCO8-pose dataset. The [COCO8-pose dataset](https://docs.ultralytics.com/datasets/pose/coco8-pose/) is a small sample dataset that's perfect for testing and debugging your pose estimation models.

Example

[Python](https://docs.ultralytics.com/tasks/pose/#python)[CLI](https://docs.ultralytics.com/tasks/pose/#cli)

```
from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n-pose.yaml")  # build a new model from YAML
model = YOLO("yolo26n-pose.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo26n-pose.yaml").load("yolo26n-pose.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="coco8-pose.yaml", epochs=100, imgsz=640)
```

### Dataset format

YOLO pose dataset format can be found in detail in the [Dataset Guide](https://docs.ultralytics.com/datasets/pose/). To convert your existing dataset from other formats (like [COCO](https://docs.ultralytics.com/datasets/pose/coco/) etc.) to YOLO format, please use the [JSON2YOLO](https://github.com/ultralytics/JSON2YOLO) tool by Ultralytics.

For custom pose estimation tasks, you can also explore specialized datasets like [Tiger-Pose](https://docs.ultralytics.com/datasets/pose/tiger-pose/) for animal pose estimation, [Hand Keypoints](https://docs.ultralytics.com/datasets/pose/hand-keypoints/) for hand tracking, or [Dog-Pose](https://docs.ultralytics.com/datasets/pose/dog-pose/) for canine pose analysis.

## Val

Validate trained YOLO26n-pose model [accuracy](https://www.ultralytics.com/glossary/accuracy) on the COCO8-pose dataset. No arguments are needed as the `model` retains its training `data` and arguments as model attributes.

Example

[Python](https://docs.ultralytics.com/tasks/pose/#python_1)[CLI](https://docs.ultralytics.com/tasks/pose/#cli_1)

```
from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n-pose.pt")  # load an official model
model = YOLO("path/to/best.pt")  # load a custom model

# Validate the model
metrics = model.val()  # no arguments needed, dataset and settings remembered
metrics.box.map  # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps  # a list containing mAP50-95 for each category
metrics.pose.map  # map50-95(P)
metrics.pose.map50  # map50(P)
metrics.pose.map75  # map75(P)
metrics.pose.maps  # a list containing mAP50-95(P) for each category
```

## Predict

Use a trained YOLO26n-pose model to run predictions on images. The [predict mode](https://docs.ultralytics.com/modes/predict/) allows you to perform inference on images, videos, or real-time streams.

Example

[Python](https://docs.ultralytics.com/tasks/pose/#python_2)[CLI](https://docs.ultralytics.com/tasks/pose/#cli_2)

```
from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n-pose.pt")  # load an official model
model = YOLO("path/to/best.pt")  # load a custom model

# Predict with the model
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

# Access the results
for result in results:
    xy = result.keypoints.xy  # x and y coordinates
    xyn = result.keypoints.xyn  # normalized
    kpts = result.keypoints.data  # x, y, visibility (if available)
```

See full `predict` mode details in the [Predict](https://docs.ultralytics.com/modes/predict/) page.

## Export

Export a YOLO26n Pose model to a different format like ONNX, CoreML, etc. This allows you to deploy your model on various platforms and devices for [real-time inference](https://www.ultralytics.com/glossary/real-time-inference).

Example

[Python](https://docs.ultralytics.com/tasks/pose/#python_3)[CLI](https://docs.ultralytics.com/tasks/pose/#cli_3)

```
from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n-pose.pt")  # load an official model
model = YOLO("path/to/best.pt")  # load a custom-trained model

# Export the model
model.export(format="onnx")
```

Available YOLO26-pose export formats are in the table below. You can export to any format using the `format` argument, i.e., `format='onnx'` or `format='engine'`. You can predict or validate directly on exported models, i.e., `yolo predict model=yolo26n-pose.onnx`. Usage examples are shown for your model after export completes.

| Format                                                       | `format` Argument | Model                            | Metadata | Arguments                                                    |
| :----------------------------------------------------------- | :---------------- | :------------------------------- | :------- | :----------------------------------------------------------- |
| [PyTorch](https://pytorch.org/)                              | -                 | `yolo26n-pose.pt`                | ✅        | -                                                            |
| [TorchScript](https://docs.ultralytics.com/integrations/torchscript/) | `torchscript`     | `yolo26n-pose.torchscript`       | ✅        | `imgsz`, `half`, `dynamic`, `optimize`, `nms`, `batch`, `device` |
| [ONNX](https://docs.ultralytics.com/integrations/onnx/)      | `onnx`            | `yolo26n-pose.onnx`              | ✅        | `imgsz`, `half`, `dynamic`, `simplify`, `opset`, `nms`, `batch`, `device` |
| [OpenVINO](https://docs.ultralytics.com/integrations/openvino/) | `openvino`        | `yolo26n-pose_openvino_model/`   | ✅        | `imgsz`, `half`, `dynamic`, `int8`, `nms`, `batch`, `data`, `fraction`, `device` |
| [TensorRT](https://docs.ultralytics.com/integrations/tensorrt/) | `engine`          | `yolo26n-pose.engine`            | ✅        | `imgsz`, `half`, `dynamic`, `simplify`, `workspace`, `int8`, `nms`, `batch`, `data`, `fraction`, `device` |
| [CoreML](https://docs.ultralytics.com/integrations/coreml/)  | `coreml`          | `yolo26n-pose.mlpackage`         | ✅        | `imgsz`, `dynamic`, `half`, `int8`, `nms`, `batch`, `device` |
| [TF SavedModel](https://docs.ultralytics.com/integrations/tf-savedmodel/) | `saved_model`     | `yolo26n-pose_saved_model/`      | ✅        | `imgsz`, `keras`, `int8`, `nms`, `batch`, `device`           |
| [TF GraphDef](https://docs.ultralytics.com/integrations/tf-graphdef/) | `pb`              | `yolo26n-pose.pb`                | ❌        | `imgsz`, `batch`, `device`                                   |
| [TF Lite](https://docs.ultralytics.com/integrations/tflite/) | `tflite`          | `yolo26n-pose.tflite`            | ✅        | `imgsz`, `half`, `int8`, `nms`, `batch`, `data`, `fraction`, `device` |
| [TF Edge TPU](https://docs.ultralytics.com/integrations/edge-tpu/) | `edgetpu`         | `yolo26n-pose_edgetpu.tflite`    | ✅        | `imgsz`, `device`                                            |
| [TF.js](https://docs.ultralytics.com/integrations/tfjs/)     | `tfjs`            | `yolo26n-pose_web_model/`        | ✅        | `imgsz`, `half`, `int8`, `nms`, `batch`, `device`            |
| [PaddlePaddle](https://docs.ultralytics.com/integrations/paddlepaddle/) | `paddle`          | `yolo26n-pose_paddle_model/`     | ✅        | `imgsz`, `batch`, `device`                                   |
| [MNN](https://docs.ultralytics.com/integrations/mnn/)        | `mnn`             | `yolo26n-pose.mnn`               | ✅        | `imgsz`, `batch`, `int8`, `half`, `device`                   |
| [NCNN](https://docs.ultralytics.com/integrations/ncnn/)      | `ncnn`            | `yolo26n-pose_ncnn_model/`       | ✅        | `imgsz`, `half`, `batch`, `device`                           |
| [IMX500](https://docs.ultralytics.com/integrations/sony-imx500/) | `imx`             | `yolo26n-pose_imx_model/`        | ✅        | `imgsz`, `int8`, `data`, `fraction`, `device`                |
| [RKNN](https://docs.ultralytics.com/integrations/rockchip-rknn/) | `rknn`            | `yolo26n-pose_rknn_model/`       | ✅        | `imgsz`, `batch`, `name`, `device`                           |
| [ExecuTorch](https://docs.ultralytics.com/integrations/executorch/) | `executorch`      | `yolo26n-pose_executorch_model/` | ✅        | `imgsz`, `device`                                            |
| [Axelera](https://docs.ultralytics.com/integrations/axelera/) | `axelera`         | `yolo26n-pose_axelera_model/`    | ✅        | `imgsz`, `int8`, `data`, `fraction`, `device`                |

See full `export` details in the [Export](https://docs.ultralytics.com/modes/export/) page.

## FAQ

### What is Pose Estimation with Ultralytics YOLO26 and how does it work?

Pose estimation with Ultralytics YOLO26 involves identifying specific points, known as keypoints, in an image. These keypoints typically represent joints or other important features of the object. The output includes the `[x, y]` coordinates and confidence scores for each point. YOLO26-pose models are specifically designed for this task and use the `-pose` suffix, such as `yolo26n-pose.pt`. These models are pretrained on datasets like [COCO keypoints](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco-pose.yaml) and can be used for various pose estimation tasks. For more information, visit the [Pose Estimation Page](https://docs.ultralytics.com/tasks/pose/#pose-estimation).

### How can I train a YOLO26-pose model on a custom dataset?

Training a YOLO26-pose model on a custom dataset involves loading a model, either a new model defined by a YAML file or a pretrained model. You can then start the training process using your specified dataset and parameters.

```
from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n-pose.yaml")  # build a new model from YAML
model = YOLO("yolo26n-pose.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="your-dataset.yaml", epochs=100, imgsz=640)
```

For comprehensive details on training, refer to the [Train Section](https://docs.ultralytics.com/tasks/pose/#train). You can also use [Ultralytics Platform](https://platform.ultralytics.com/) for a no-code approach to training custom pose estimation models.

### How do I validate a trained YOLO26-pose model?

Validation of a YOLO26-pose model involves assessing its accuracy using the same dataset parameters retained during training. Here's an example:

```
from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n-pose.pt")  # load an official model
model = YOLO("path/to/best.pt")  # load a custom model

# Validate the model
metrics = model.val()  # no arguments needed, dataset and settings remembered
```

For more information, visit the [Val Section](https://docs.ultralytics.com/tasks/pose/#val).

### Can I export a YOLO26-pose model to other formats, and how?

Yes, you can export a YOLO26-pose model to various formats like ONNX, CoreML, TensorRT, and more. This can be done using either Python or the Command Line Interface (CLI).

```
from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n-pose.pt")  # load an official model
model = YOLO("path/to/best.pt")  # load a custom-trained model

# Export the model
model.export(format="onnx")
```

Refer to the [Export Section](https://docs.ultralytics.com/tasks/pose/#export) for more details. Exported models can be deployed on edge devices for [real-time applications](https://www.ultralytics.com/blog/real-time-inferences-in-vision-ai-solutions-are-making-an-impact) like fitness tracking, sports analysis, or [robotics](https://www.ultralytics.com/blog/from-algorithms-to-automation-ais-role-in-robotics).

### What are the available Ultralytics YOLO26-pose models and their performance metrics?

Ultralytics YOLO26 offers various pretrained pose models such as YOLO26n-pose, YOLO26s-pose, YOLO26m-pose, among others. These models differ in size, accuracy (mAP), and speed. For instance, the YOLO26n-pose model achieves a mAPpose50-95 of 50.0 and an mAPpose50 of 81.0. For a complete list and performance details, visit the [Models Section](https://docs.ultralytics.com/tasks/pose/#models).