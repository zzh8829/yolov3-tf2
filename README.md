# YoloV3 Implemented in Tensorflow 2.0

This repo provides a clean implementation of YoloV3 in Tensorflow 2.0 using all the best practices.

## Key Features

- [x] Tensorflow 2.0
- [x] `yolov3` with pre-trained Weights
- [ ] `yolov3-tiny` with pre-trained Weights
- [x] Inference example
- [ ] Transfer learning example
- [ ] Training from scratch example
- [x] Eager training with `tf.GradientTape`
- [x] Functional model with `tf.keras.layers`
- [x] Input pipeline using `tf.data`
- [x] Vectorized transformations
- [ ] GPU accelerated
- [x] Fully integrated with `absl-py` abseil.io
- [x] Clean implementation
- [x] Following the best practices
- [x] MIT License

## Usage

### Installation
```
pip install -r requirements.txt
```

### Detection

```
python detect.py
```

### Training

```
python train.py
```

### Convert Darknet weights

```
python convert.py
```

## Implementation Details

### Eager execution

Great addition for existing Tensorflow experts.
Not very easy to use without some intermediate understanding of Tensorflow graphs.

### @tf.function

@tf.function is very cool. Do have some caveats tho.

### Loading pre-trained Darknet weights

very hard without compromising the model structure.

### tf.keras.layers.BatchNormalization

It doesn't work very well for transfer learning

## References

It is pretty much impossible to implement this from the yolov3 paper alone. I had to reference the official (very hard to understand) and many un-official (many minor errors) repos to piece together the complete picture.

- https://github.com/pjreddie/darknet
    - official yolov3 implementation
- https://github.com/AlexeyAB
    - explinations of parameters
- https://github.com/qqwweee/keras-yolo3
    - models
    - loss functions
- https://github.com/YunYang1994/tensorflow-yolov3
    - data transformations
    - loss functions
- https://github.com/ayooshkathuria/pytorch-yolo-v3
    - models
- https://github.com/broadinstitute/keras-resnet
    - batch normalization fix
