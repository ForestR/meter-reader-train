# mega-meter-reader-train

[本项目](https://github.com/ForestR/mega-meter-reader-train)用于基于 **YOLO26** 的读数检测模型后训练，采用三阶段模块化管道。

| 阶段 | 任务 | 基座模型 |
|------|------|----------|
| 1 | 表盘区域分割（mask） | `yolo26n-seg.pt` |
| 2 | 数字检测（bbox） | `yolo26n.pt` |
| 3 | 数字分类 | `yolo26n-cls.pt` |

---

## Stage 1：表盘分割

**标注**：segment（像素级 mask），两类标签：

- **dial**：完整读数区域  
- **decimal_section**：小数位区域（小数点后）

**训练配置**

- `nc=2`  
- **输入**：RAW 图像  
- **输出**：dial 与 decimal_section 的 mask  

---

## Stage 1 后处理：DIAL ROI

**目的**

1. 显式标出小数区域（语义信息）。  
2. 利用 dial 的长方形 mask 与 decimal_section 的相对位置，确定阅读方向。

真实业务图像常有角度偏转与镜头畸变。为降低后续阶段难度，需将 **dial ∪ decimal_section** 并集所对应的近似矩形区域 **长轴对齐到水平**，并使 **decimal_section 位于 dial 右侧**（从左到右阅读，小数位在末端）。

**几何对齐（实现）**

- 在 **axis-aligned union crop** 上，对并集二值 mask 用 **`cv2.PCACompute2`** 的特征值得到各向异性 `sqrt(λ_major / λ_minor)`，用于 **长细比阈值**（默认低于 2.0 视为退化，与正方形类形状一并拒绝）。  
- **旋转角** 使用中心协方差的 **图像朝向**公式 `0.5 * atan2(2·σ_xy, σ_xx − σ_yy)`（度，落在 (-90°, 90°]），与 OpenCV `getRotationMatrix2D` 一致；避免在 `σ_xx ≈ σ_yy` 时误把「最大方差方向」当成几何长轴。  
- 旋转与扩画布后，根据 **warp 后的 dial / decimal 质心**：若 decimal 在 dial **左侧**，再作 **180°** 翻转以固定阅读方向。

**数据流**

- **输入**：RAW 图像 + dial、decimal_section 的 mask  
- **处理概要**：并集轴对齐 bbox（加 margin）crop → 上述 PCA + 朝向角旋转 crop → **按旋转后并集轮廓**（`findContours` + `transform`）定紧致轴对齐外包，再按 **宽、高分别** 乘 `margin_ratio` 加 padding（避免细长目标被「取长边 padding」撑爆短边）→ 背景绿色填充轮廓外 → 必要时 180° 翻转  
- **输出**：**DIAL_ROI** 图像与 2×3 仿射 `affine_matrix`（与 `cv2.warpAffine` 作用于整图 mask 时所用矩阵一致）

**校验与异常**

- 每组输入须 **同时** 含 **有且仅有一个** dial 与 **有且仅有一个** decimal_section；否则告警并 **跳过 Stage 2、Stage 3**，结果中标记 `is_invalid=True`。  
- dial 与 decimal_section **必须有交集**；否则同样告警、跳过后续阶段，`is_invalid=True`。  
- 并集 mask **点数过少**、**PCA 长细比低于阈值**，或 **两质心重合** 时：`build_dial_roi` 返回 `is_invalid=True` 并附 `warnings`，仿射退化为仅平移 crop。

### 单元测试：DIAL ROI 长轴对齐

长轴角度与 OpenCV 符号约定易在实现中反复出错，因此用 **合成旋转矩形** 做回归测试（不依赖真实标注）：

- **路径**：[`tests/test_stage1_pca_alignment.py`](../tests/test_stage1_pca_alignment.py)  
- **内容**：水平细长矩形绕画布中心旋转 `{0°, 30°, 45°, 60°, 90°}` 后，调用 `pca_long_axis_alignment` 再 `warpAffine`，断言二次对齐所需转角接近 0（容许误差约 2°）；另含 **正方形退化**（期望 `ValueError`）与 **点数过少** 用例。  
- **运行**（需 `pip install -e ".[dev]"`，`pyproject.toml` 已配置 `pythonpath = ["src"]`）：

```bash
pytest tests/test_stage1_pca_alignment.py -v
```

GT mask 可视化整链可继续用 [`scripts/test_postprocess_gt.py`](../scripts/test_postprocess_gt.py)（`--data` / `--split` / `--out`）。

### 单元测试：DIAL ROI 画布紧致度

[`tests/test_stage1_dial_roi_canvas.py`](../tests/test_stage1_dial_roi_canvas.py) 用合成 dial/dec mask 调用 `build_dial_roi`，将全图 union mask 用输出仿射做 `warpAffine` 得到 ROI 内前景 tight bbox，断言 **输出宽高分别不超过前景 bbox 宽高的 150%**（加少量舍入容差）。

```bash
pytest tests/test_stage1_dial_roi_canvas.py -v
```

**备注（实现中曾失败或弃用的路径及原因）**

- **用 dial / decimal 质心连线 `atan2` 定主旋转**：只反映两区域的相对位置，不保证与并集几何长轴一致；分割抖动时方向不稳。已改为由 **union mask** 估计朝向。  
- **用 PCA 第一特征向量的 `atan2` 作旋转角**：当 `σ_xx ≈ σ_yy`（例如细长条约在 45°）时，**最大特征值对应的方向可以是短边**（方差在垂直于长条方向更大），与几何长轴可差 90°；合成单测在 30°/45°/60° 失败。现改用中心协方差的 **图像朝向** `0.5 * atan2(2σ_xy, σ_xx − σ_yy)`。  
- **在两条 PCA 轴上比「投影跨度」选手动长轴 + 配合 `-atan2` 与特征向量符号**：仍易与 OpenCV 旋转符号纠缠；已统一到上述二阶矩朝向公式。  
- **输出尺寸按旋转后「整幅 crop」轴对齐外包计算**：并集往往只占 crop 一部分，画布相对内容过松。现改为对 **union 外轮廓点**经同一 `getRotationMatrix2D` 矩阵变换后取紧 **AABB**。  
- **padding 用 `max(宽, 高) * margin_ratio` 同时扩宽高**：对细长水平目标会把 **短边** 撑得过大，无法通过「画布边长 ≤ 前景 bbox 边长 ×150%」类断言。现改为 **宽、高分别** 用 `margin_ratio` 得 `pad_x` / `pad_y`。  
- **测试里对 `affine_matrix` 求逆再 `warpAffine` 全图 mask**：与当前实现中传给 `warpAffine` 的 2×3 矩阵用法不一致，会得到 **空 warped union**。测试中应 **直接** 使用 `Stage1Output.affine_matrix`。  
- **`cv2.minAreaRect` 主对齐路径**：可作调试对照，未并入主流程，以降低双路径维护成本。

---

## Stage 2：数字检测

在 DIAL_ROI 内检测每个数字的位置。

- `nc=1`  
- **输入**：DIAL_ROI 图像  
- **输出**：数字 bounding boxes  
- **说明**：本阶段只负责在给定区域内提取所有数字的 bbox。

---

## Stage 3：数字分类

对 Stage 2 给出的每个数字区域做类别预测（读数识别）。

- **基座模型**：`yolo26n-cls.pt`  
- `nc=20`：全字符 `{0,1,…,9}` 与半字符（中间态）`{0-1,1-2,…,8-9,9-0}`  
- **输入**：数字 bbox 对应图像块（如 64×64）  
- **输出**：每个位置的分类结果  

---

## Stage 3 后处理：排序与小数点

1. **水平排序**：按 Stage 2 中 bbox 的 **从左到右** 顺序排列 Stage 3 的字符结果。  

   示例：`"1,2,3,4,5,6"`、`"9,5,8,8a,2,6,6,1a"`、`"0,0,8,2,3a,9,8,5"`

2. **半字符语义**：`{digit}a` 表示该位处于 `{digit}` 与下一数字之间的中间态，语义上近似「半格」，例如 `3a` ≈ 3 与 4 之间（约 3.5）。

3. **整数 / 小数划分**：根据 **仿射变换后** decimal_section 的位置，将排序后的序列分为整数段与小数段。  

   示例：

   - `"1,2,3,4,[5,6]"` → `"1234.56"`  
   - `"9,5,8,8a,2,[6,6,1a]"` → `"95882.661"`  
   - `"0,0,8,2,3a,9,[8,5]"` → `"008239.85"`

4. **半字符输出规则**：含 `{digit}a` 的位 **向前取整**（`{digit}a` → `{digit}`），并在结果中标记 `is_half_digit=True`。

---

## 输出 JSON

| 字段 | 类型 | 含义 |
|------|------|------|
| `value` | string | 读数字符串 |
| `is_half_digit` | bool | 是否存在半字符位（按上述规则取整后仍须记录） |
| `is_invalid` | bool | 是否为无效输入（Stage 1 校验失败等） |

---

## 数据与配置

- **Stage 1 标注格式**：见 [`data/stage1_dial_seg/INFO.md`](data/stage1_dial_seg/INFO.md)（Ultralytics NDJSON segment 说明）。  
- **训练 mix 配置**：`datasets/mix_stage1_v1.yaml` 等。

## 架构文档

更细的架构设计（训练 + 推理、概念层）见 [`docs/01_overview.md`](docs/01_overview.md) 及同目录下 `02`–`06` 各篇。

---

## Stage 1 代码（实现）

环境建议使用 Conda `meter-reader-turbo`（或已安装 PyTorch + Ultralytics 的环境）。依赖见 [`requirements-stage1.txt`](requirements-stage1.txt)。

```bash
cd /path/to/mega-meter-reader-train
pip install -e ".[dev]"
pytest -q
# DIAL ROI 长轴对齐（建议 CI / 改 postprocess 后必跑）：
pytest tests/test_stage1_pca_alignment.py -v
pytest tests/test_stage1_dial_roi_canvas.py -v
```

**训练**（单 NDJSON 或合并 mix；默认基座为可自动下载的 `yolo11n-seg.pt`，若有 `yolo26n-seg.pt` 可 `--model` 指定本地路径）：

```bash
python -m mega_meter_reader.stage1.train \
  --data data/stage1_dial_seg/meter-panel.ndjson \
  --epochs 100 --imgsz 640 --batch 8

python -m mega_meter_reader.stage1.train \
  --mix datasets/mix_stage1_v1.yaml \
  --epochs 100

python -m mega_meter_reader.stage1.train \
  --config configs/mega_meter_reader/stage1/train.yaml
```

`--config` 从 YAML 读取 `mix`/`data`、`model` 以及 Ultralytics 训练参数；命令行中显式传入的选项会覆盖配置文件。`mix` 模式会把 `train_policy` 中多个 NDJSON **按行合并**为临时文件再训练（`weight` 暂不用于采样）。权重默认写入 `runs/stage1_seg/train/weights/best.pt`。

快速冒烟（1 epoch、子采样数据，需可访问 NDJSON 中的图像 URL 或本地已缓存）：

```bash
python -m mega_meter_reader.stage1.train \
  --data data/stage1_dial_seg/meter-panel.ndjson \
  --epochs 1 --batch 2 --imgsz 320 --fraction 0.03 \
  --project runs/stage1_seg_smoke --name smoke
```

**推理 + DIAL ROI**（需分割权重）：

```bash
python -m mega_meter_reader.stage1.infer_cli \
  --weights runs/stage1_seg/train/weights/best.pt \
  --source path/to/image.jpg \
  --out artifacts/dial_roi.jpg
```

Python API：`mega_meter_reader.stage1.predict.run_stage1(model, image_bgr)`，返回 [`Stage1Output`](src/mega_meter_reader/stage1/types.py) 与 Ultralytics `Results`。
