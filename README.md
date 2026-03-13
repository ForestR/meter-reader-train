# Project Plan: YOLO-based Meter Reading Fine-tuning Framework

## 1. Project Scope & Positioning

**Core Focus:**
This repository focuses on the **downstream fine-tuning phase** of YOLO models for edge-based meter reading (OCR). We utilize Transfer Learning, initializing from official upstream weights (e.g., YOLO11, YOLOv8/26) to adapt to specific industrial scenarios.

**Out of Scope:**
*   Pre-training backbone networks on ImageNet/COCO.
*   Developing novel detection architectures (we strictly use YOLO-based architectures).
*   Natural Language Processing (NLP) based post-correction.

---

## 2. Model Architecture

We support two deployment topologies: **End-to-End** and **3-Stage Pipeline**.

### 2.1 Topologies

*   **End2End Model:**
    *   **Input:** Raw snapshot.
    *   **Output:** Meter reading sequence directly.
    *   **Use Case:** High-performance edge devices; scenarios with consistent lighting and low skew.
*   **Pipeline Model (Recommended for Robustness):**
    *   **Stage 1 - Localization (Dial-Det):** Detects the meter display area. *Optional: Includes keypoints for alignment.*
    *   **Stage 2 - Segmentation/Detection (Digit-Det):** Detects individual digit regions within the aligned dial.
    *   **Stage 3 - Recognition (Digit-Cls):** Classifies the specific number (0-9, half-digits) for each region.

### 2.2 Configuration Management
*Refactored to separate Model Topology from Runtime settings.*

**`configs/model_topology.yaml`**
```yaml
# Model Architecture Definition
MODEL_TYPE: pipeline # 'end2end' or 'pipeline'

# Path Definitions (Renamed for semantic accuracy)
CHECKPOINTS:
  END2END: checkpoints/production/end2end_yolo11l.pt
  PIPELINE:
    STAGE_1_DIAL: checkpoints/production/01_dial_det.pt  # Formerly 'seg', renamed to 'det'
    STAGE_2_DIGIT: checkpoints/production/02_digit_det.pt # Formerly 'seg', renamed to 'det'
    STAGE_3_CLS: checkpoints/production/03_digit_cls.pt
```

**`configs/runtime_policy.yaml`**
```yaml
# Inference & Deployment Settings
DEVICE: auto
BATCH_SIZE: 32
CONFIDENCE_THRESHOLDS:
  DIAL: 0.8
  DIGIT: 0.6 # Lower threshold allowed inside a verified dial
IMAGE_Process:
  MAX_SIZE_MB: 10
  TARGET_Resolution: [800, 600]
```

---

## 3. Technical Decision: Angle Correction strategy

**Problem:** Edge sensors produce images with horizontal skew (≤ ±15°).
**Decision:** **Image Alignment + Standard Box Detection**.

| Component | Strategy | Rationale |
| :--- | :--- | :--- |
| **Model Type** | **Standard YOLO (Horizontal Box)** | OBB (Rotated Box) is over-engineered for discrete digits. Digits are invariant to rotation in topology, but order matters. |
| **Pre-processing** | **Geometric Alignment** | Stage 1 (Dial-Det) will define the ROI. We apply a perspective transform or rotation to "straighten" the dial before passing it to Stage 2. |
| **Data Immutability** | **View-Only Correction** | We do not overwrite raw datasets with rotated versions. Correction happens on-the-fly during training (augmentation) and inference (pre-processing). |

**Upgrade Path:**
*   *Future Optimization:* Upgrade Stage 1 from `yolo-det` to `yolo-pose`. Detecting the 4 corners of the screen provides a perfect homography matrix for rectification, superior to simple bounding boxes.

---

## 4. Training Strategy

We adopt a **phased unfreezing strategy** to balance stability and plasticity.

### 4.1 Phased Training Protocol
1.  **Phase 1: Head Adaptation (Freeze Backbone)**
    *   **Action:** Freeze all backbone layers (CSP/ELAN modules). Train only the Head and Neck.
    *   **Goal:** Rapidly adapt the detection head to the new class definitions (e.g., "meter_dial") without destroying pretrained feature extractors.
    *   **Epochs:** ~20-50.
2.  **Phase 2: Deep Fine-tuning (Unfreeze All)**
    *   **Action:** Unfreeze full model. Apply **LLRD (Layer-wise Learning Rate Decay)**—lower LR for early layers, higher LR for head.
    *   **Goal:** Allow the backbone to learn domain-specific textures (LCD reflections, mechanical wheel fonts).
    *   **Hyperparams:** LR = 10% of Phase 1.

### 4.2 Pipeline Specifics
*   **Stage 1 (Dial):** Often works well with just Phase 1 (Transfer learning is easy for large objects).
*   **Stage 3 (Digits):** Requires Phase 2 (Fine-tuning) as specific fonts vary significantly between manufacturers.

---

## 5. Data Engineering: The "Dataset Manifest"

We move from simple folder scanning to a **Manifest-based Data Loader**. This allows "Physical Isolation, Logical Mixing."

### 5.1 Physical Storage (Source of Truth)
Datasets are stored essentially immutable, separated by collection source.
```text
data/
 ├─ raw_xuzhou_2023/       (City specific)
 ├─ raw_changzhou_2024/    (City specific)
 ├─ raw_negative_samples/  (Pure background, walls, pipes - Critical for reducing False Positives)
 ├─ raw_hard_cases/        (Manually curated failures)
```

### 5.2 Logical Mixing (Manifests)
We define training sets using YAML files, enabling reproducible experiments without duplicating data.

**`datasets/mix_v1_robust.yaml`**
```yaml
# Training Mix for General Robustness
train_policy:
  - source: data/raw_xuzhou_2023
    weight: 1.0
  - source: data/raw_changzhou_2024
    weight: 1.0
  - source: data/raw_negative_samples # The "Dirty" Dataset
    weight: 0.2 # 20% of batch will be background to teach model what is NOT a meter
    label_map: empty # Images have no boxes
  - source: data/raw_hard_cases
    weight: 2.0 # Oversample hard cases
```

---

## 6. MLOps: Data Annotation Flywheel

To handle the influx of edge data, we implement a **Human-in-the-Loop** workflow.

### 6.1 The 3-Tier Pipeline
1.  **Tier 1: Auto-Labeling (Pre-annotation)**
    *   Use the best current `Pipeline Model` to predict labels on new data.
    *   Generate `.xml` or `.txt` labels automatically.
2.  **Tier 2: Active Learning Filter (The "Smart" Step)**
    *   **Confidence Filtering:** Flag images where model confidence is $0.4 < conf < 0.7$ (Ambiguous).
    *   **Logic Check:** Flag images where Stage 2 detects 5 digits but Stage 3 recognizes 4.
3.  **Tier 3: Human Verification**
    *   Annotators only review the flagged "Tier 2" images and "Tier 1" random samples (QA).
    *   Tools: Label Studio / CVAT.

### 6.2 Handling "Hard Samples"
*   Images that fail in production (reported by users or logic checks) are added to `data/raw_hard_cases`.
*   The `mix_v1_robust.yaml` is updated to oversample this folder in the next training cycle.

---

## Summary of Refactor Changes (vs Original)

1.  **Terminology Fix:** Renamed `seg` (segmentation) checkpoints to `det` (detection) to match standard YOLO nomenclature and avoid confusion.
2.  **Config Structure:** Split configuration into `Model` (Architecture) and `Runtime` (Deployment) for better MLOps scaling.
3.  **Dataset Strategy:** Formalized the "Dirty Dataset" idea into a **Manifest System**. This allows explicit weighting of negative samples and oversampling of hard cases.
4.  **Training Detail:** Explicitly defined the "Freeze -> Unfreeze" phases and added LLRD as a best practice.
5.  **Fail-Safe:** Implicitly added logical checks (e.g., in the Active Learning section) to catch pipeline failures.