# Object Detection

![YOLO object detection with bounding boxes](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/object-detection-examples.avif)

[Object detection](https://www.ultralytics.com/glossary/object-detection) is a task that involves identifying the location and class of objects in an image or video stream.

The output of an object detector is a set of bounding boxes that enclose the objects in the image, along with class labels and confidence scores for each box. Object detection is a good choice when you need to identify objects of interest in a scene, but don't need to know exactly where the object is or its exact shape.



**Watch:** Object Detection with Pretrained Ultralytics YOLO Model.

Tip

YOLO26 Detect models are the default YOLO26 models, i.e., `yolo26n.pt`, and are pretrained on [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml).

## [Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/26)

YOLO26 pretrained Detect models are shown here. Detect, Segment, and Pose models are pretrained on the [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) dataset, while Classify models are pretrained on the [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml) dataset.

[Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) are downloaded automatically from the latest Ultralytics [release](https://github.com/ultralytics/assets/releases) on first use.

| Model                                                        | size (pixels) | mAPval 50-95 | mAPval 50-95(e2e) | Speed CPU ONNX (ms) | Speed T4 TensorRT10 (ms) | params (M) | FLOPs (B) |
| :----------------------------------------------------------- | :------------ | :----------- | :---------------- | :------------------ | :----------------------- | :--------- | :-------- |
| [YOLO26n](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt) | 640           | 40.9         | 40.1              | 38.9 ± 0.7          | 1.7 ± 0.0                | 2.4        | 5.4       |
| [YOLO26s](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt) | 640           | 48.6         | 47.8              | 87.2 ± 0.9          | 2.5 ± 0.0                | 9.5        | 20.7      |
| [YOLO26m](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt) | 640           | 53.1         | 52.5              | 220.0 ± 1.4         | 4.7 ± 0.1                | 20.4       | 68.2      |
| [YOLO26l](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l.pt) | 640           | 55.0         | 54.4              | 286.2 ± 2.0         | 6.2 ± 0.2                | 24.8       | 86.4      |
| [YOLO26x](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x.pt) | 640           | 57.5         | 56.9              | 525.8 ± 4.0         | 11.8 ± 0.2               | 55.7       | 193.9     |

- **mAPval** values are for single-model single-scale on [COCO val2017](https://cocodataset.org/) dataset.
  Reproduce by `yolo val detect data=coco.yaml device=0`
- **Speed** averaged over COCO val images using an [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) instance.
  Reproduce by `yolo val detect data=coco.yaml batch=1 device=0|cpu`

## Train

Train YOLO26n on the COCO8 dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) at image size 640. For a full list of available arguments see the [Configuration](https://docs.ultralytics.com/usage/cfg/) page.

Example

[Python](https://docs.ultralytics.com/tasks/detect/#python)[CLI](https://docs.ultralytics.com/tasks/detect/#cli)

```
from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n.yaml")  # build a new model from YAML
model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo26n.yaml").load("yolo26n.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
```

### Dataset format

YOLO detection dataset format can be found in detail in the [Dataset Guide](https://docs.ultralytics.com/datasets/detect/). To convert your existing dataset from other formats (like COCO etc.) to YOLO format, please use [JSON2YOLO](https://github.com/ultralytics/JSON2YOLO) tool by Ultralytics.

## Val

Validate trained YOLO26n model [accuracy](https://www.ultralytics.com/glossary/accuracy) on the COCO8 dataset. No arguments are needed as the `model` retains its training `data` and arguments as model attributes.

Example

[Python](https://docs.ultralytics.com/tasks/detect/#python_1)[CLI](https://docs.ultralytics.com/tasks/detect/#cli_1)

```
from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n.pt")  # load an official model
model = YOLO("path/to/best.pt")  # load a custom model

# Validate the model
metrics = model.val()  # no arguments needed, dataset and settings remembered
metrics.box.map  # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps  # a list containing mAP50-95 for each category
```

## Predict

Use a trained YOLO26n model to run predictions on images.

Example

[Python](https://docs.ultralytics.com/tasks/detect/#python_2)[CLI](https://docs.ultralytics.com/tasks/detect/#cli_2)

```
from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n.pt")  # load an official model
model = YOLO("path/to/best.pt")  # load a custom model

# Predict with the model
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

# Access the results
for result in results:
    xywh = result.boxes.xywh  # center-x, center-y, width, height
    xywhn = result.boxes.xywhn  # normalized
    xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
    xyxyn = result.boxes.xyxyn  # normalized
    names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
    confs = result.boxes.conf  # confidence score of each box
```

See full `predict` mode details in the [Predict](https://docs.ultralytics.com/modes/predict/) page.

## Export

Export a YOLO26n model to a different format like ONNX, CoreML, etc.

Example

[Python](https://docs.ultralytics.com/tasks/detect/#python_3)[CLI](https://docs.ultralytics.com/tasks/detect/#cli_3)

```
from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n.pt")  # load an official model
model = YOLO("path/to/best.pt")  # load a custom-trained model

# Export the model
model.export(format="onnx")
```

Available YOLO26 export formats are in the table below. You can export to any format using the `format` argument, i.e., `format='onnx'` or `format='engine'`. You can predict or validate directly on exported models, i.e., `yolo predict model=yolo26n.onnx`. Usage examples are shown for your model after export completes.

| Format                                                       | `format` Argument | Model                       | Metadata | Arguments                                                    |
| :----------------------------------------------------------- | :---------------- | :-------------------------- | :------- | :----------------------------------------------------------- |
| [PyTorch](https://pytorch.org/)                              | -                 | `yolo26n.pt`                | ✅        | -                                                            |
| [TorchScript](https://docs.ultralytics.com/integrations/torchscript/) | `torchscript`     | `yolo26n.torchscript`       | ✅        | `imgsz`, `half`, `dynamic`, `optimize`, `nms`, `batch`, `device` |
| [ONNX](https://docs.ultralytics.com/integrations/onnx/)      | `onnx`            | `yolo26n.onnx`              | ✅        | `imgsz`, `half`, `dynamic`, `simplify`, `opset`, `nms`, `batch`, `device` |
| [OpenVINO](https://docs.ultralytics.com/integrations/openvino/) | `openvino`        | `yolo26n_openvino_model/`   | ✅        | `imgsz`, `half`, `dynamic`, `int8`, `nms`, `batch`, `data`, `fraction`, `device` |
| [TensorRT](https://docs.ultralytics.com/integrations/tensorrt/) | `engine`          | `yolo26n.engine`            | ✅        | `imgsz`, `half`, `dynamic`, `simplify`, `workspace`, `int8`, `nms`, `batch`, `data`, `fraction`, `device` |
| [CoreML](https://docs.ultralytics.com/integrations/coreml/)  | `coreml`          | `yolo26n.mlpackage`         | ✅        | `imgsz`, `dynamic`, `half`, `int8`, `nms`, `batch`, `device` |
| [TF SavedModel](https://docs.ultralytics.com/integrations/tf-savedmodel/) | `saved_model`     | `yolo26n_saved_model/`      | ✅        | `imgsz`, `keras`, `int8`, `nms`, `batch`, `device`           |
| [TF GraphDef](https://docs.ultralytics.com/integrations/tf-graphdef/) | `pb`              | `yolo26n.pb`                | ❌        | `imgsz`, `batch`, `device`                                   |
| [TF Lite](https://docs.ultralytics.com/integrations/tflite/) | `tflite`          | `yolo26n.tflite`            | ✅        | `imgsz`, `half`, `int8`, `nms`, `batch`, `data`, `fraction`, `device` |
| [TF Edge TPU](https://docs.ultralytics.com/integrations/edge-tpu/) | `edgetpu`         | `yolo26n_edgetpu.tflite`    | ✅        | `imgsz`, `device`                                            |
| [TF.js](https://docs.ultralytics.com/integrations/tfjs/)     | `tfjs`            | `yolo26n_web_model/`        | ✅        | `imgsz`, `half`, `int8`, `nms`, `batch`, `device`            |
| [PaddlePaddle](https://docs.ultralytics.com/integrations/paddlepaddle/) | `paddle`          | `yolo26n_paddle_model/`     | ✅        | `imgsz`, `batch`, `device`                                   |
| [MNN](https://docs.ultralytics.com/integrations/mnn/)        | `mnn`             | `yolo26n.mnn`               | ✅        | `imgsz`, `batch`, `int8`, `half`, `device`                   |
| [NCNN](https://docs.ultralytics.com/integrations/ncnn/)      | `ncnn`            | `yolo26n_ncnn_model/`       | ✅        | `imgsz`, `half`, `batch`, `device`                           |
| [IMX500](https://docs.ultralytics.com/integrations/sony-imx500/) | `imx`             | `yolo26n_imx_model/`        | ✅        | `imgsz`, `int8`, `data`, `fraction`, `device`                |
| [RKNN](https://docs.ultralytics.com/integrations/rockchip-rknn/) | `rknn`            | `yolo26n_rknn_model/`       | ✅        | `imgsz`, `batch`, `name`, `device`                           |
| [ExecuTorch](https://docs.ultralytics.com/integrations/executorch/) | `executorch`      | `yolo26n_executorch_model/` | ✅        | `imgsz`, `device`                                            |
| [Axelera](https://docs.ultralytics.com/integrations/axelera/) | `axelera`         | `yolo26n_axelera_model/`    | ✅        | `imgsz`, `int8`, `data`, `fraction`, `device`                |

See full `export` details in the [Export](https://docs.ultralytics.com/modes/export/) page.

## FAQ

### How do I train a YOLO26 model on my custom dataset?

Training a YOLO26 model on a custom dataset involves a few steps:

1. **Prepare the Dataset**: Ensure your dataset is in the YOLO format. For guidance, refer to our [Dataset Guide](https://docs.ultralytics.com/datasets/detect/).
2. **Load the Model**: Use the Ultralytics YOLO library to load a pretrained model or create a new model from a YAML file.
3. **Train the Model**: Execute the `train` method in Python or the `yolo detect train` command in CLI.

Example

[Python](https://docs.ultralytics.com/tasks/detect/#python_4)[CLI](https://docs.ultralytics.com/tasks/detect/#cli_4)

```
from ultralytics import YOLO

# Load a pretrained model
model = YOLO("yolo26n.pt")

# Train the model on your custom dataset
model.train(data="my_custom_dataset.yaml", epochs=100, imgsz=640)
```

For detailed configuration options, visit the [Configuration](https://docs.ultralytics.com/usage/cfg/) page.

### What pretrained models are available in YOLO26?

Ultralytics YOLO26 offers various pretrained models for object detection, segmentation, and pose estimation. These models are pretrained on the COCO dataset or ImageNet for classification tasks. Here are some of the available models:

- [YOLO26n](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt)
- [YOLO26s](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt)
- [YOLO26m](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt)
- [YOLO26l](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l.pt)
- [YOLO26x](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x.pt)

For a detailed list and performance metrics, refer to the [Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/26) section.

### How can I validate the accuracy of my trained YOLO model?

To validate the accuracy of your trained YOLO26 model, you can use the `.val()` method in Python or the `yolo detect val` command in CLI. This will provide metrics like mAP50-95, mAP50, and more.

Example

[Python](https://docs.ultralytics.com/tasks/detect/#python_5)[CLI](https://docs.ultralytics.com/tasks/detect/#cli_5)

```
from ultralytics import YOLO

# Load the model
model = YOLO("path/to/best.pt")

# Validate the model
metrics = model.val()
print(metrics.box.map)  # mAP50-95
```

For more validation details, visit the [Val](https://docs.ultralytics.com/modes/val/) page.

### What formats can I export a YOLO26 model to?

Ultralytics YOLO26 allows exporting models to various formats such as [ONNX](https://www.ultralytics.com/glossary/onnx-open-neural-network-exchange), [TensorRT](https://www.ultralytics.com/glossary/tensorrt), [CoreML](https://docs.ultralytics.com/integrations/coreml/), and more to ensure compatibility across different platforms and devices.

Example

[Python](https://docs.ultralytics.com/tasks/detect/#python_6)[CLI](https://docs.ultralytics.com/tasks/detect/#cli_6)

```
from ultralytics import YOLO

# Load the model
model = YOLO("yolo26n.pt")

# Export the model to ONNX format
model.export(format="onnx")
```

Check the full list of supported formats and instructions on the [Export](https://docs.ultralytics.com/modes/export/) page.

### Why should I use Ultralytics YOLO26 for object detection?

Ultralytics YOLO26 is designed to offer state-of-the-art performance for object detection, segmentation, and pose estimation. Here are some key advantages:

1. **Pretrained Models**: Utilize models pretrained on popular datasets like [COCO](https://docs.ultralytics.com/datasets/detect/coco/) and [ImageNet](https://docs.ultralytics.com/datasets/classify/imagenet/) for faster development.
2. **High Accuracy**: Achieves impressive mAP scores, ensuring reliable object detection.
3. **Speed**: Optimized for [real-time inference](https://www.ultralytics.com/glossary/real-time-inference), making it ideal for applications requiring swift processing.
4. **Flexibility**: Export models to various formats like ONNX and TensorRT for deployment across multiple platforms.

Explore our [Blog](https://www.ultralytics.com/blog) for use cases and success stories showcasing YOLO26 in action.