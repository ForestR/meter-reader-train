## Ultralytics NDJSON format

[Ultralytics NDJSON format](https://docs.ultralytics.com/datasets/detect/#segment)

The NDJSON (Newline Delimited JSON) format provides an alternative way to define datasets for Ultralytics YOLO models. This format stores dataset metadata and annotations in a single file where each line contains a separate JSON object.

An NDJSON dataset file contains:

+ Dataset record (first line): Contains dataset metadata including task type, class names, and general information
+ Image records (subsequent lines): Contains individual image data including dimensions, annotations, and file paths

### NDJSON Example

Segment:

```json
{
    "type": "image",
    "file": "image1.jpg",
    "url": "https://www.url.com/path/to/image1.jpg",
    "width": 640,
    "height": 480,
    "split": "train",
    "annotations": {
        "segments": [
            [0, 0.681, 0.485, 0.670, 0.487, 0.676, 0.487, 0.688, 0.515],
            [1, 0.422, 0.315, 0.438, 0.330, 0.445, 0.328, 0.450, 0.320]
        ]
    }
}
```
Format: `[class_id, x1, y1, x2, y2, x3, y3, ...]`


### Usage Example

To use an NDJSON dataset with YOLO26, simply specify the path to the .ndjson file:

```Python
from ultralytics import YOLO

# Load a segment model (Stage 1 uses *-seg.pt, not detect yolo26n.pt)
model = YOLO("yolo26n-seg.pt")

# Train using NDJSON dataset
results = model.train(data="path/to/dataset.ndjson", epochs=100, imgsz=640)
```

### Advantages of NDJSON format

+ Single file: All dataset information contained in one file
+ Streaming: Can process large datasets line-by-line without loading everything into memory
+ Cloud integration: Supports remote image URLs for cloud-based training
+ Extensible: Easy to add custom metadata fields
+ Version control: Single file format works well with git and version control systems
