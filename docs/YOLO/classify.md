# Image Classification

![YOLO image classification of objects and scenes](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/image-classification-examples.avif)

[Image classification](https://www.ultralytics.com/glossary/image-classification) is the simplest of the three tasks and involves classifying an entire image into one of a set of predefined classes.

The output of an image classifier is a single class label and a confidence score. Image classification is useful when you need to know only what class an image belongs to and don't need to know where objects of that class are located or what their exact shape is.

**Watch:** Explore Ultralytics YOLO Tasks: Image Classification using Ultralytics Platform

Tip

YOLO26 Classify models use the `-cls` suffix, i.e., `yolo26n-cls.pt`, and are pretrained on [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml).

## [Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/26)

YOLO26 pretrained Classify models are shown here. Detect, Segment, and Pose models are pretrained on the [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) dataset, while Classify models are pretrained on the [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml) dataset.

[Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) are downloaded automatically from the latest Ultralytics [release](https://github.com/ultralytics/assets/releases) on first use.

| Model                                                        | size (pixels) | acc top1 | acc top5 | Speed CPU ONNX (ms) | Speed T4 TensorRT10 (ms) | params (M) | FLOPs (B) at 224 |
| :----------------------------------------------------------- | :------------ | :------- | :------- | :------------------ | :----------------------- | :--------- | :--------------- |
| [YOLO26n-cls](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-cls.pt) | 224           | 71.4     | 90.1     | 5.0 ± 0.3           | 1.1 ± 0.0                | 2.8        | 0.5              |
| [YOLO26s-cls](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-cls.pt) | 224           | 76.0     | 92.9     | 7.9 ± 0.2           | 1.3 ± 0.0                | 6.7        | 1.6              |
| [YOLO26m-cls](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m-cls.pt) | 224           | 78.1     | 94.2     | 17.2 ± 0.4          | 2.0 ± 0.0                | 11.6       | 4.9              |
| [YOLO26l-cls](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l-cls.pt) | 224           | 79.0     | 94.6     | 23.2 ± 0.3          | 2.8 ± 0.0                | 14.1       | 6.2              |
| [YOLO26x-cls](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x-cls.pt) | 224           | 79.9     | 95.0     | 41.4 ± 0.9          | 3.8 ± 0.0                | 29.6       | 13.6             |

- **acc** values are model accuracies on the [ImageNet](https://www.image-net.org/) dataset validation set.
  Reproduce by `yolo val classify data=path/to/ImageNet device=0`
- **Speed** averaged over ImageNet val images using an [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) instance.
  Reproduce by `yolo val classify data=path/to/ImageNet batch=1 device=0|cpu`

## Train

Train YOLO26n-cls on the MNIST160 dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) at image size 64. For a full list of available arguments see the [Configuration](https://docs.ultralytics.com/usage/cfg/) page.

Example

[Python](https://docs.ultralytics.com/tasks/classify/#python)[CLI](https://docs.ultralytics.com/tasks/classify/#cli)

```
from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n-cls.yaml")  # build a new model from YAML
model = YOLO("yolo26n-cls.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo26n-cls.yaml").load("yolo26n-cls.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="mnist160", epochs=100, imgsz=64)
```

Tip

Ultralytics YOLO classification uses [`torchvision.transforms.RandomResizedCrop`](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.RandomResizedCrop.html) for training and [`torchvision.transforms.CenterCrop`](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.CenterCrop.html) for validation and inference. These cropping-based transforms assume square inputs and may inadvertently crop out important regions from images with extreme aspect ratios, potentially causing loss of critical visual information during training. To preserve the full image while maintaining its proportions, consider using [`torchvision.transforms.Resize`](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.Resize.html) instead of cropping transforms.

You can implement this by customizing your augmentation pipeline through a custom `ClassificationDataset` and `ClassificationTrainer`.

```
import torch
import torchvision.transforms as T

from ultralytics import YOLO
from ultralytics.data.dataset import ClassificationDataset
from ultralytics.models.yolo.classify import ClassificationTrainer, ClassificationValidator


class CustomizedDataset(ClassificationDataset):
    """A customized dataset class for image classification with enhanced data augmentation transforms."""

    def __init__(self, root: str, args, augment: bool = False, prefix: str = ""):
        """Initialize a customized classification dataset with enhanced data augmentation transforms."""
        super().__init__(root, args, augment, prefix)

        # Add your custom training transforms here
        train_transforms = T.Compose(
            [
                T.Resize((args.imgsz, args.imgsz)),
                T.RandomHorizontalFlip(p=args.fliplr),
                T.RandomVerticalFlip(p=args.flipud),
                T.RandAugment(interpolation=T.InterpolationMode.BILINEAR),
                T.ColorJitter(brightness=args.hsv_v, contrast=args.hsv_v, saturation=args.hsv_s, hue=args.hsv_h),
                T.ToTensor(),
                T.Normalize(mean=torch.tensor(0), std=torch.tensor(1)),
                T.RandomErasing(p=args.erasing, inplace=True),
            ]
        )

        # Add your custom validation transforms here
        val_transforms = T.Compose(
            [
                T.Resize((args.imgsz, args.imgsz)),
                T.ToTensor(),
                T.Normalize(mean=torch.tensor(0), std=torch.tensor(1)),
            ]
        )
        self.torch_transforms = train_transforms if augment else val_transforms


class CustomizedTrainer(ClassificationTrainer):
    """A customized trainer class for YOLO classification models with enhanced dataset handling."""

    def build_dataset(self, img_path: str, mode: str = "train", batch=None):
        """Build a customized dataset for classification training and the validation during training."""
        return CustomizedDataset(root=img_path, args=self.args, augment=mode == "train", prefix=mode)


class CustomizedValidator(ClassificationValidator):
    """A customized validator class for YOLO classification models with enhanced dataset handling."""

    def build_dataset(self, img_path: str, mode: str = "train"):
        """Build a customized dataset for classification standalone validation."""
        return CustomizedDataset(root=img_path, args=self.args, augment=mode == "train", prefix=self.args.split)


model = YOLO("yolo26n-cls.pt")
model.train(data="imagenet1000", trainer=CustomizedTrainer, epochs=10, imgsz=224, batch=64)
model.val(data="imagenet1000", validator=CustomizedValidator, imgsz=224, batch=64)
```

### Dataset format

YOLO classification dataset format can be found in detail in the [Dataset Guide](https://docs.ultralytics.com/datasets/classify/).

## Val

Validate trained YOLO26n-cls model [accuracy](https://www.ultralytics.com/glossary/accuracy) on the MNIST160 dataset. No arguments are needed as the `model` retains its training `data` and arguments as model attributes.

Example

[Python](https://docs.ultralytics.com/tasks/classify/#python_1)[CLI](https://docs.ultralytics.com/tasks/classify/#cli_1)

```
from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n-cls.pt")  # load an official model
model = YOLO("path/to/best.pt")  # load a custom model

# Validate the model
metrics = model.val()  # no arguments needed, dataset and settings remembered
metrics.top1  # top1 accuracy
metrics.top5  # top5 accuracy
```

Tip

As mentioned in the [training section](https://docs.ultralytics.com/tasks/classify/#train), you can handle extreme aspect ratios during training by using a custom `ClassificationTrainer`. You need to apply the same approach for consistent validation results by implementing a custom `ClassificationValidator` when calling the `val()` method. Refer to the complete code example in the [training section](https://docs.ultralytics.com/tasks/classify/#train) for implementation details.

## Predict

Use a trained YOLO26n-cls model to run predictions on images.

Example

[Python](https://docs.ultralytics.com/tasks/classify/#python_2)[CLI](https://docs.ultralytics.com/tasks/classify/#cli_2)

```
from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n-cls.pt")  # load an official model
model = YOLO("path/to/best.pt")  # load a custom model

# Predict with the model
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
```

See full `predict` mode details in the [Predict](https://docs.ultralytics.com/modes/predict/) page.

## Export

Export a YOLO26n-cls model to a different format like ONNX, CoreML, etc.

Example

[Python](https://docs.ultralytics.com/tasks/classify/#python_3)[CLI](https://docs.ultralytics.com/tasks/classify/#cli_3)

```
from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n-cls.pt")  # load an official model
model = YOLO("path/to/best.pt")  # load a custom-trained model

# Export the model
model.export(format="onnx")
```

Available YOLO26-cls export formats are in the table below. You can export to any format using the `format` argument, i.e., `format='onnx'` or `format='engine'`. You can predict or validate directly on exported models, i.e., `yolo predict model=yolo26n-cls.onnx`. Usage examples are shown for your model after export completes.

| Format                                                       | `format` Argument | Model                           | Metadata | Arguments                                                    |
| :----------------------------------------------------------- | :---------------- | :------------------------------ | :------- | :----------------------------------------------------------- |
| [PyTorch](https://pytorch.org/)                              | -                 | `yolo26n-cls.pt`                | ✅        | -                                                            |
| [TorchScript](https://docs.ultralytics.com/integrations/torchscript/) | `torchscript`     | `yolo26n-cls.torchscript`       | ✅        | `imgsz`, `half`, `dynamic`, `optimize`, `nms`, `batch`, `device` |
| [ONNX](https://docs.ultralytics.com/integrations/onnx/)      | `onnx`            | `yolo26n-cls.onnx`              | ✅        | `imgsz`, `half`, `dynamic`, `simplify`, `opset`, `nms`, `batch`, `device` |
| [OpenVINO](https://docs.ultralytics.com/integrations/openvino/) | `openvino`        | `yolo26n-cls_openvino_model/`   | ✅        | `imgsz`, `half`, `dynamic`, `int8`, `nms`, `batch`, `data`, `fraction`, `device` |
| [TensorRT](https://docs.ultralytics.com/integrations/tensorrt/) | `engine`          | `yolo26n-cls.engine`            | ✅        | `imgsz`, `half`, `dynamic`, `simplify`, `workspace`, `int8`, `nms`, `batch`, `data`, `fraction`, `device` |
| [CoreML](https://docs.ultralytics.com/integrations/coreml/)  | `coreml`          | `yolo26n-cls.mlpackage`         | ✅        | `imgsz`, `dynamic`, `half`, `int8`, `nms`, `batch`, `device` |
| [TF SavedModel](https://docs.ultralytics.com/integrations/tf-savedmodel/) | `saved_model`     | `yolo26n-cls_saved_model/`      | ✅        | `imgsz`, `keras`, `int8`, `nms`, `batch`, `device`           |
| [TF GraphDef](https://docs.ultralytics.com/integrations/tf-graphdef/) | `pb`              | `yolo26n-cls.pb`                | ❌        | `imgsz`, `batch`, `device`                                   |
| [TF Lite](https://docs.ultralytics.com/integrations/tflite/) | `tflite`          | `yolo26n-cls.tflite`            | ✅        | `imgsz`, `half`, `int8`, `nms`, `batch`, `data`, `fraction`, `device` |
| [TF Edge TPU](https://docs.ultralytics.com/integrations/edge-tpu/) | `edgetpu`         | `yolo26n-cls_edgetpu.tflite`    | ✅        | `imgsz`, `device`                                            |
| [TF.js](https://docs.ultralytics.com/integrations/tfjs/)     | `tfjs`            | `yolo26n-cls_web_model/`        | ✅        | `imgsz`, `half`, `int8`, `nms`, `batch`, `device`            |
| [PaddlePaddle](https://docs.ultralytics.com/integrations/paddlepaddle/) | `paddle`          | `yolo26n-cls_paddle_model/`     | ✅        | `imgsz`, `batch`, `device`                                   |
| [MNN](https://docs.ultralytics.com/integrations/mnn/)        | `mnn`             | `yolo26n-cls.mnn`               | ✅        | `imgsz`, `batch`, `int8`, `half`, `device`                   |
| [NCNN](https://docs.ultralytics.com/integrations/ncnn/)      | `ncnn`            | `yolo26n-cls_ncnn_model/`       | ✅        | `imgsz`, `half`, `batch`, `device`                           |
| [IMX500](https://docs.ultralytics.com/integrations/sony-imx500/) | `imx`             | `yolo26n-cls_imx_model/`        | ✅        | `imgsz`, `int8`, `data`, `fraction`, `device`                |
| [RKNN](https://docs.ultralytics.com/integrations/rockchip-rknn/) | `rknn`            | `yolo26n-cls_rknn_model/`       | ✅        | `imgsz`, `batch`, `name`, `device`                           |
| [ExecuTorch](https://docs.ultralytics.com/integrations/executorch/) | `executorch`      | `yolo26n-cls_executorch_model/` | ✅        | `imgsz`, `device`                                            |
| [Axelera](https://docs.ultralytics.com/integrations/axelera/) | `axelera`         | `yolo26n-cls_axelera_model/`    | ✅        | `imgsz`, `int8`, `data`, `fraction`, `device`                |

See full `export` details in the [Export](https://docs.ultralytics.com/modes/export/) page.

## FAQ

### What is the purpose of YOLO26 in image classification?

YOLO26 models, such as `yolo26n-cls.pt`, are designed for efficient image classification. They assign a single class label to an entire image along with a confidence score. This is particularly useful for applications where knowing the specific class of an image is sufficient, rather than identifying the location or shape of objects within the image.

### How do I train a YOLO26 model for image classification?

To train a YOLO26 model, you can use either Python or CLI commands. For example, to train a `yolo26n-cls` model on the MNIST160 dataset for 100 epochs at an image size of 64:

Example

[Python](https://docs.ultralytics.com/tasks/classify/#python_4)[CLI](https://docs.ultralytics.com/tasks/classify/#cli_4)

```
from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n-cls.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="mnist160", epochs=100, imgsz=64)
```

For more configuration options, visit the [Configuration](https://docs.ultralytics.com/usage/cfg/) page.

### Where can I find pretrained YOLO26 classification models?

Pretrained YOLO26 classification models can be found in the [Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/26) section. Models like `yolo26n-cls.pt`, `yolo26s-cls.pt`, `yolo26m-cls.pt`, etc., are pretrained on the [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml) dataset and can be easily downloaded and used for various image classification tasks.

### How can I export a trained YOLO26 model to different formats?

You can export a trained YOLO26 model to various formats using Python or CLI commands. For instance, to export a model to ONNX format:

Example

[Python](https://docs.ultralytics.com/tasks/classify/#python_5)[CLI](https://docs.ultralytics.com/tasks/classify/#cli_5)

```
from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n-cls.pt")  # load the trained model

# Export the model to ONNX
model.export(format="onnx")
```

For detailed export options, refer to the [Export](https://docs.ultralytics.com/modes/export/) page.

### How do I validate a trained YOLO26 classification model?

To validate a trained model's accuracy on a dataset like MNIST160, you can use the following Python or CLI commands:

Example

[Python](https://docs.ultralytics.com/tasks/classify/#python_6)[CLI](https://docs.ultralytics.com/tasks/classify/#cli_6)

```
from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n-cls.pt")  # load the trained model

# Validate the model
metrics = model.val()  # no arguments needed, uses the dataset and settings from training
metrics.top1  # top1 accuracy
metrics.top5  # top5 accuracy
```

For more information, visit the [Validate](https://docs.ultralytics.com/tasks/classify/#val) section.