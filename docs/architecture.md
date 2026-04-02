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

真实业务图像常有角度偏转与镜头畸变。为降低后续阶段难度，需将 dial 对应的长方形 mask **旋转到水平**，并使 **decimal_section 位于长方形右侧**（从左到右阅读，小数位在末端）。

**实现思路（可选）**

- 用图像矩等方法提取 dial 长方形 mask 的长轴、短轴；处理后长轴对齐水平、短轴对齐竖直。  
- 用两 mask 质心构造从「主体」指向「尾部」的参考向量，修正主轴朝向（正负号）。

**数据流**

- **输入**：RAW 图像 + dial、decimal_section 的 mask  
- **处理概要**：基于 mask（并集）求最小外接矩形 → 按 mask crop → 计算方向向量并旋转 crop → 计算旋转后画布尺寸 → 自适应填充（如背景绿色）→ 将原 crop 置于画布中心 → 以画布中心为旋转中心做仿射变换  
- **输出**：**DIAL_ROI** 图像（物体水平居中、四周 padding、无内容丢失）

**校验与异常**

- 每组输入须 **同时** 含 **有且仅有一个** dial 与 **有且仅有一个** decimal_section；否则告警并 **跳过 Stage 2、Stage 3**，结果中标记 `is_invalid=True`。  
- dial 与 decimal_section **必须有交集**；否则同样告警、跳过后续阶段，`is_invalid=True`。

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
