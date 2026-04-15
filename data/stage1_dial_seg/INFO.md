# Stage 1 NDJSON datasets (dial segmentation)

Ultralytics Platform exports **NDJSON** files (see [docs/repository/YOLO/ndjson.md](../../docs/repository/YOLO/ndjson.md)).

## Classes (`nc = 2`)

| ID | Name              | Role                                      |
|----|-------------------|-------------------------------------------|
| 0  | `dial`            | Full dial / readout region (mask)         |
| 1  | `decimal_section` | Decimal-part region (mask)                |

Task: **segment** (not detect).

## Files in this directory

- `meter-panel.ndjson` — meter panel samples  
- `water-meter.ndjson` — water meter samples  

First line of each file is dataset metadata (`type: dataset`, `class_names`, …).  
Following lines are `type: image` records with optional `annotations.segments`.

## Materialized layout (cache)

Training code materializes each NDJSON next to the file:

```text
data/stage1_dial_seg/<ndjson_stem>/
  images/train/  images/val/  [images/test/]
  labels/train/  labels/val/  [labels/test/]
  data.yaml
```

A **mix** of several NDJSON sources uses symlinks under `datasets/mix_stage1_v1/` (see [yolo_ndjson_symlink_mix_and_cache.md](../../docs/repository/YOLO/yolo_ndjson_symlink_mix_and_cache.md)).

## Training entrypoints

```bash
python -m mega_meter_reader.stage1.train --data data/stage1_dial_seg/meter-panel.ndjson
python -m mega_meter_reader.stage1.train --mix datasets/mix_stage1_v1.yaml
python -m mega_meter_reader.stage1.train --config configs/mega_meter_reader/stage1/train.yaml
```

Use `--refresh-data` to delete cached materialization and rebuild.
