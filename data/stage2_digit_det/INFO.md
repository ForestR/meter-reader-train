# Stage 2 digit detection data (`data/stage2_digit_det/`)

## Scope: single source of truth

**Training and evaluation for Stage-2 digit detection are defined by the NDJSON files in this directory** (human-annotated and reviewed on Ultralytics Platform). Do not treat ad-hoc local YOLO trees or auto-label outputs as authoritative unless they are reproducibly exported from these NDJSONs.

Authoritative files (names are fixed):

| Dataset | NDJSON |
|--------|--------|
| Meter panel (dial ROI) | [`meter-panel-cropped-dial.ndjson`](meter-panel-cropped-dial.ndjson) |
| Water meter (dial ROI) | [`water-meter-cropped-dial.ndjson`](water-meter-cropped-dial.ndjson) |

- **Task:** YOLO **detect**, `nc=1`, class `0` = `digit`.
- **Images:** **dial ROI crops** (same pixel domain as Stage-2 training). Splits (`train` / `val` / `test`) are embedded per image in the NDJSON (`split` field).

---

## Data provenance: how dial ROI crops were produced

The **images** in these datasets are dial-region crops. A typical pipeline to obtain such crops from Stage-1 data is:

1. Start from a Stage-1 segmentation dataset (full panel images + YOLO **segment** labels: dial + decimal_section polygons).
2. Apply Stage-1 **post-processing** (mask validation + dial ROI extraction) to produce a cropped dial image per sample.

For a concrete, reproducible check of that post-process on **ground-truth** Stage-1 masks, see [`scripts/test_postprocess_gt.py`](../../scripts/test_postprocess_gt.py) (it reads `images/<split>` and `labels/<split>`, rasterizes masks, then writes DIAL ROI JPEGs). Example (provenance / smoke test only—not the NDJSON authoring step):

```bash
python scripts/test_postprocess_gt.py \
  --data data/stage1_dial_seg/meter-panel \
  --split test \
  --out results/test_postprocess_gt/
```

The **digit bounding boxes** in the Stage-2 NDJSON are **not** produced by that script; they come from manual annotation / review on top of the dial ROI domain.

---

## NDJSON schema

### Dataset header (first line, JSON object)

- `type`: `"dataset"`
- `task`: `"detect"`
- `name`, `description`, `url`, `class_names`, `version`, timestamps, etc.
- `class_names`: e.g. `{"0":"digit"}` → single class, id `0`.

### Image records (following lines, one JSON object per line)

Each record has `type: "image"` and typically:

| Field | Meaning |
|-------|---------|
| `file` | Image filename (basename). |
| `url` | CDN URL for the image bytes. |
| `width`, `height` | Image size in pixels. |
| `split` | One of `train`, `val`, `test`. |
| `annotations.boxes` | List of detect boxes for this image. |

**Box format** (YOLO detect, normalized 0–1):

Each box is `[class_id, x_center, y_center, width, height]` with coordinates relative to image width/height.

Example: `[0, 0.5, 0.5, 0.1, 0.2]` is class `digit`, centered, with relative size 10% × 20% of the image.

### Downstream classification (not Stage-2 detect)

The dataset `description` in the NDJSON may note that **digit classification** downstream uses 20 semantic categories (digits + transitions such as `9-0`). That affects **cls** design; Stage-2 **detect** here still uses a single class `digit`. A single box may sometimes cover two digit positions in edge cases—see the in-file description.

---

## Naming conventions (avoid confusion)

| What | Convention in this doc |
|------|-------------------------|
| **NDJSON files** | Keep existing names: `meter-panel-cropped-dial.ndjson`, `water-meter-cropped-dial.ndjson`. |
| **Optional local YOLO tree** (if you materialize to disk) | Use a **different** directory name from the NDJSON stem, e.g. `data/stage2_digit_det/meter-panel_yolo/` or `data/stage2_digit_det/water-meter_yolo/`. The `*_yolo/` suffix is a naming suggestion only; **the NDJSON remains the source of truth**. |

---

## Recommended workflow (NDJSON-first)

1. Treat the NDJSON as the **only** canonical definition of images, splits, and labels.
2. Any export (YOLO directory layout, caches, re-splits) should be **derivable from** and **traceable to** the NDJSON (e.g. same MD5 / version).
3. **Auto-label** outputs are **not** a substitute for the reviewed NDJSON when reporting Stage-2 training data.

---

## Optional: pre-label / bootstrap (not the training ground truth)

These tools can help **bootstrap** labels on dial crops (e.g. before manual fix), but they do **not** replace the human-reviewed NDJSON above.

### Why not `demo_pipeline.py --auto-label stage2` on dial crops?

That path runs **Stage 1 on the full image**, then crops the dial and runs Stage 2. If the input is **already** a dial crop, running Stage 1 again can misalign boxes. For crops-only auto-labeling, use **`scripts/auto_label_stage2_crops.py`** (Stage-2 detector only on each image).

### Example: auto-label flat `images/` + `labels/`, then split

From repo root, with `configs/model_topology.yaml` pointing to Stage-2 weights (`STAGE_2_DIGIT`), you might use a **flat** tree under a distinct folder (not the NDJSON stem), e.g.:

```bash
python scripts/auto_label_stage2_crops.py \
  --src data/stage2_digit_det/_flat_meter_panel_crops \
  --dst data/stage2_digit_det/_tmp_meter_panel_labeled
```

```bash
python scripts/split_yolo_flat_to_splits.py \
  --src data/stage2_digit_det/_tmp_meter_panel_labeled \
  --dst data/stage2_digit_det/meter-panel_yolo \
  --train 0.7 --val 0.15 --seed 42 \
  --name "meter panel stage2 (auto-label example)"
```

- Remaining fraction goes to **test** (e.g. 0.7 + 0.15 → 0.15 test).
- Default is **copy**; `--move` removes files from `--src` after splitting.
- `--allow-missing-labels` — include images with no `.txt`.

Repeat for water meter with a separate `--dst`, e.g. `data/stage2_digit_det/water-meter_yolo`.

### Optional YOLO layout (when exporting for Ultralytics)

```text
<dataset>/
  data.yaml
  images/train  images/val  images/test
  labels/train  labels/val  labels/test
```

Point Ultralytics / `train_pipeline_stage2.py` at `<dataset>/data.yaml` when training from that tree.

---

## Optional: NDJSON → local YOLO tree (reference only)

Stage-1 code includes NDJSON parsing and materialization in [`src/mega_meter_reader/stage1/dataset.py`](../../src/mega_meter_reader/stage1/dataset.py) (`materialize_ndjson`, caching by NDJSON MD5, etc.). You may reuse the **ideas** (parse header + image lines, download by URL, write `images/<split>/` and `labels/<split>/`).

**Important:** That module’s fallback path targets **`task: segment`** and writes segment-style labels from `annotations.segments`. **Stage-2 NDJSON is `task: detect`** with `annotations.boxes`. Do **not** copy it blindly as the official Stage-2 exporter; implement or use a **detect**-aware export for `boxes`, or Ultralytics’ own NDJSON conversion if applicable.

---

## Quick checklist

- [ ] Training data identity is taken from **`meter-panel-cropped-dial.ndjson` / `water-meter-cropped-dial.ndjson`**, not from ad-hoc folders alone.
- [ ] Dial ROI provenance is understood (Stage-1 GT + post-process; see `test_postprocess_gt.py` for a GT mask → ROI example).
- [ ] Labels are **detect** boxes: `[cls, xc, yc, w, h]` normalized; `0` = digit.
- [ ] Optional local trees use names like **`*_yolo/`** to avoid clashing with NDJSON stems.
