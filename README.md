# YOLOv3-TF2.x

YOLOv3 implementation in TensorFlow 2.x

## Installation

```
pip install yolov3-tf2
```

> Depends on tensorflow >=2.3.0 <=2.9.1

## Usage

The package consists of three core modules -

- dataset
- models
- utils

### Dataset

The `dataset.py` module is for loading and transforming the tfrecords for object detection. The examples in the input tfrecords must match the parsing schema.

```python
import yolov3_tf2.dataset as dataset
train_dataset = dataset.load_tfrecord_dataset(tfrecords_path)
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.map(
    lambda x, y: (
        dataset.transform_images(x, image_dim),
        dataset.transform_targets(y, anchors, anchor_masks, image_dim),
    )
)
```

### Models

The `models.py` module consists of implementation of two YOLOv3 and YOLOv3 tiny in Tesnsorflow.

```python
from yolov3_tf2.models import YoloV3, YoloV3Tiny
model = YoloV3(image_dim = 416, training=True, classes=10)
```

### Utils

The `utils.py` module provides some common functions for training YOLOv3 model, viz., loading weights, freezing layers, drawing boxes on images, compute iou

```python
# convert weights 
from yolov3_tf2.models import YoloV3, YoloV3Tiny
from yolov3_tf2 import utils

yolo = YoloV3()
utils.load_darknet_weights(yolo, weights_path, is_tiny=False)
yolo.save_weights(converted_weights_path)
```