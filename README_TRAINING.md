# Training Guide: End-to-End Meter Reading Model

This guide walks you through training a YOLO11-based end-to-end meter reading model using the phased training strategy.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Data Preparation](#data-preparation)
- [Training Process](#training-process)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### Hardware Requirements

- **GPU**: NVIDIA GPU with at least 12GB VRAM (recommended for YOLO11-Large)
  - For smaller GPUs (8GB), reduce batch size to 8 or use YOLO11-Medium
- **RAM**: At least 16GB system RAM
- **Storage**: 20GB+ free space for datasets, checkpoints, and logs

### Software Requirements

1. **Python 3.8+**
2. **CUDA 11.0+** (for GPU training)
3. **PyTorch 2.0+**

### Installation

1. Clone the repository and navigate to the project root:

```bash
cd /backup/d.yin/meter-reader-train
```

2. Create a virtual environment (recommended):

```bash
# Using conda
conda create -n meter-reader python=3.10
conda activate meter-reader

# OR using venv
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# venv\Scripts\activate  # On Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Verify installation:

```bash
python -c "from ultralytics import YOLO; print('✓ Ultralytics installed')"
python -c "import torch; print(f'✓ PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})')"
```

## Data Preparation

### 1. Data Structure

Your dataset must follow YOLO format. Each data source should have this structure:

```
data/raw_xuzhou_2023/
├── images/
│   ├── meter_001.jpg
│   ├── meter_002.jpg
│   └── ...
└── labels/
    ├── meter_001.txt
    ├── meter_002.txt
    └── ...
```

**YOLO Label Format** (`meter_001.txt`):
```
<class_id> <x_center> <y_center> <width> <height>
```

Where:
- `class_id`: Integer class ID (0 for meter_display)
- `x_center`, `y_center`: Normalized center coordinates (0-1)
- `width`, `height`: Normalized box dimensions (0-1)

Example:
```
0 0.5 0.45 0.8 0.6
```

### 2. Create Data Directories

Run the preparation script to create the directory structure:

```bash
python scripts/prepare_data.py --create-structure
```

This creates:
```
data/
├── raw_xuzhou_2023/       # City-specific datasets
├── raw_changzhou_2024/
├── raw_negative_samples/  # Background images (no meters)
└── raw_hard_cases/        # Manually curated failures
```

### 3. Add Your Data

Copy your images and labels into the appropriate directories:

```bash
# Example: copying xuzhou data
cp /path/to/xuzhou/images/* data/raw_xuzhou_2023/images/
cp /path/to/xuzhou/labels/* data/raw_xuzhou_2023/labels/

# Negative samples (only images, no labels)
cp /path/to/background/images/* data/raw_negative_samples/images/
```

### 4. Validate Data

Before training, validate your data structure:

```bash
python scripts/prepare_data.py --manifest datasets/mix_v1_robust.yaml
```

Expected output:
```
✓ Manifest validation PASSED
  Total images: 1000
  Total labels: 1000
  Weighted dataset size: 1500
```

### 5. Configure Manifest (Optional)

Edit `datasets/mix_v1_robust.yaml` to adjust data source weights:

```yaml
train_policy:
  - source: data/raw_xuzhou_2023
    weight: 1.0  # Standard weight
  - source: data/raw_changzhou_2024
    weight: 1.0
  - source: data/raw_negative_samples
    weight: 0.2  # 20% negative samples to reduce false positives
    label_map: empty
  - source: data/raw_hard_cases
    weight: 2.0  # Oversample hard cases
```

**Weight interpretation:**
- `weight: 1.0` = Include each image once
- `weight: 2.0` = Include each image twice (oversampling)
- `weight: 0.5` = Include each image with 50% probability

## Training Process

### Overview

Training follows a **2-phase strategy**:

1. **Phase 1 (30 epochs)**: Freeze backbone, train only detection head
   - Fast adaptation to meter reading task
   - Preserves ImageNet features
   
2. **Phase 2 (50 epochs)**: Unfreeze all layers, fine-tune with lower LR
   - Learn domain-specific features (LCD reflections, mechanical fonts)
   - 10% learning rate of Phase 1

### Quick Start Training

Run the complete training pipeline:

```bash
python src/train_end2end.py
```

This will:
1. Load the manifest and generate data configuration
2. Execute Phase 1 training (frozen backbone)
3. Execute Phase 2 training (unfrozen, fine-tuning)
4. Save checkpoints to `checkpoints/end2end/`

### Step-by-Step Training

#### Dry Run (Validation Only)

Test your configuration without training:

```bash
python src/train_end2end.py --dry-run
```

#### Phase 1 Only

Train only Phase 1 (useful for quick iterations):

```bash
python src/train_end2end.py --phase1-only
```

#### Resume from Phase 1

If Phase 1 completed successfully, manually run Phase 2:

```bash
# Edit configs/train_phase2_unfrozen.yaml
# Set resume: true if interrupted

python src/train_end2end.py
```

### Monitoring Training

#### TensorBoard

View real-time training metrics:

```bash
tensorboard --logdir runs/detect
```

Open `http://localhost:6006` in your browser.

#### Console Output

Training progress is displayed in the console:

```
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
  1/30      8.2G      1.234      0.567      0.891         32        640
  2/30      8.3G      1.123      0.543      0.876         32        640
...
```

Key metrics:
- `box_loss`: Bounding box localization loss
- `cls_loss`: Classification loss
- `dfl_loss`: Distribution focal loss (YOLO11 specific)
- `mAP50`: Mean Average Precision @ IoU=0.5 (validation)

#### Checkpoints

Checkpoints are saved to:

```
checkpoints/end2end/
├── phase1_best.pt    # Best model from Phase 1
├── phase2_best.pt    # Best model from Phase 2
└── final.pt          # Final production model
```

### Expected Results

Typical training metrics for meter reading:

| Metric | Phase 1 (Frozen) | Phase 2 (Fine-tuned) |
|--------|------------------|----------------------|
| mAP50 | 0.75-0.85 | 0.85-0.95 |
| Precision | 0.80-0.90 | 0.90-0.95 |
| Recall | 0.75-0.85 | 0.85-0.92 |

Training time (on V100):
- Phase 1: ~2-3 hours (30 epochs)
- Phase 2: ~3-4 hours (50 epochs)
- **Total**: ~5-7 hours

## Advanced Usage

### Custom Manifest

Create a custom manifest for specific scenarios:

```bash
# Create new manifest
cat > datasets/my_custom_mix.yaml << EOF
train_policy:
  - source: data/my_custom_dataset
    weight: 1.0
EOF

# Train with custom manifest
python src/train_end2end.py --manifest datasets/my_custom_mix.yaml
```

### Adjust Hyperparameters

Edit training configs to tune hyperparameters:

**For faster training** (edit `configs/train_phase1_frozen.yaml`):
```yaml
epochs: 20  # Reduce from 30
batch: 32   # Increase if GPU allows
```

**For better accuracy** (edit `configs/train_phase2_unfrozen.yaml`):
```yaml
epochs: 100  # Increase from 50
lr0: 0.0005  # Lower learning rate
```

**For smaller GPUs**:
```yaml
batch: 8     # Reduce batch size
imgsz: 512   # Reduce image size
```

### Multi-GPU Training

For multiple GPUs, use `device` parameter:

```bash
# Use GPUs 0 and 1
# Edit configs/train_phase1_frozen.yaml:
device: [0, 1]
```

### Resume Interrupted Training

If training is interrupted:

```bash
# Phase 1 interruption
python src/train_end2end.py --phase1-only

# Phase 2 interruption
# Edit configs/train_phase2_unfrozen.yaml: resume: true
python src/train_end2end.py
```

## Model Evaluation

After training, evaluate the model:

### On Validation Set

```python
from ultralytics import YOLO

# Load trained model
model = YOLO('checkpoints/end2end/final.pt')

# Evaluate
metrics = model.val()

print(f"mAP50: {metrics.box.map50:.3f}")
print(f"mAP50-95: {metrics.box.map:.3f}")
print(f"Precision: {metrics.box.p:.3f}")
print(f"Recall: {metrics.box.r:.3f}")
```

### On Test Images

```python
from ultralytics import YOLO

model = YOLO('checkpoints/end2end/final.pt')

# Predict on single image
results = model.predict('test_images/meter_001.jpg', save=True)

# Predict on directory
results = model.predict('test_images/', save=True)
```

Results are saved to `runs/detect/predict/`.

## Troubleshooting

### Issue: Out of Memory (OOM)

**Solution**: Reduce batch size and/or image size

```yaml
# Edit configs/train_phase1_frozen.yaml
batch: 8     # Reduce from 16
imgsz: 512   # Reduce from 640
```

### Issue: Poor mAP (<0.5)

**Possible causes**:
1. **Insufficient data**: Add more training images
2. **Label quality**: Validate labels with `prepare_data.py`
3. **Class imbalance**: Adjust weights in manifest
4. **Wrong image size**: Meters too small? Increase `imgsz: 800`

**Solution**: Check label quality first

```bash
python scripts/prepare_data.py --manifest datasets/mix_v1_robust.yaml
```

### Issue: Training Loss Not Decreasing

**Possible causes**:
1. **Learning rate too low**: Increase `lr0` (Phase 1: 0.01 → 0.02)
2. **Learning rate too high**: Decrease `lr0` (Phase 2: 0.001 → 0.0005)
3. **Frozen too many layers**: Reduce `freeze` parameter

**Solution**: Check learning rate and adjust

### Issue: Model Overfitting

**Symptoms**: Training loss decreases, but validation loss increases

**Solution**: Add more augmentation

```yaml
# Edit configs/train_phase1_frozen.yaml
degrees: 20.0    # Increase rotation
mosaic: 1.0      # Keep mosaic
mixup: 0.2       # Increase mixup
```

### Issue: Missing pretrained weights

**Error**: `FileNotFoundError: yolo11l.pt`

**Solution**: Download automatically on first run, or manually:

```bash
wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolo11l.pt
```

### Issue: CUDA not available

**Solution**: Install PyTorch with CUDA support

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Next Steps

After successful training:

1. **Update model topology**: Edit `config/model_topology.yaml`:
   ```yaml
   CHECKPOINTS:
     END2END: checkpoints/end2end/final.pt
   ```

2. **Export for deployment**: Convert to ONNX/TensorRT for edge devices:
   ```python
   from ultralytics import YOLO
   model = YOLO('checkpoints/end2end/final.pt')
   model.export(format='onnx')  # or 'engine' for TensorRT
   ```

3. **Implement inference pipeline**: Create deployment script for edge devices

4. **Monitor in production**: Collect failure cases to add to `data/raw_hard_cases/`

5. **Retrain periodically**: Update manifest with new hard cases and retrain

## Key Concepts

### Angle Correction Strategy

The training implements angle correction through **rotation augmentation**:

- **Training time**: Random rotation ±15° teaches model to handle skewed images
- **Augmentation**: Configured via `degrees: 15.0` in training YAML
- **Inference time**: Model handles ±15° skew automatically

For pipeline mode (future), Stage 1 will use keypoint-based geometric alignment.

### Dataset Manifest System

The manifest enables **"Physical Isolation, Logical Mixing"**:

- **Physical**: Datasets stored separately by source (city, collection date)
- **Logical**: Mixed via weighted sampling in manifest
- **Benefits**: 
  - No data duplication
  - Easy to oversample hard cases
  - Reproducible experiments
  - Version control friendly

### Phased Training Strategy

**Why 2 phases?**

1. **Stability**: Freezing backbone prevents catastrophic forgetting of ImageNet features
2. **Speed**: Phase 1 converges quickly (only training 20% of parameters)
3. **Plasticity**: Phase 2 fine-tunes for domain-specific features

This is a best practice for transfer learning in computer vision.

## Support

For issues and questions:
- Check [Troubleshooting](#troubleshooting) section
- Review Ultralytics docs: https://docs.ultralytics.com
- Check logs in `runs/detect/`

## References

- [YOLO11 Documentation](https://docs.ultralytics.com)
- [Transfer Learning Guide](https://docs.ultralytics.com/modes/train/#resuming-interrupted-trainings)
- [Data Augmentation](https://docs.ultralytics.com/modes/train/#augmentation-settings-and-hyperparameters)
