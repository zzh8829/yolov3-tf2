# YoloV3 Implemented in Tensorflow 2.0

This repo provides a clean implementation of YoloV3 in Tensorflow 2.0 using all the best practices.

## Key Features

- [x] Tensorflow 2.0
- [x] `yolov3` with pre-trained Weights
- [x] `yolov3-tiny` with pre-trained Weights
- [x] Inference example
- [ ] Transfer learning example
- [ ] Training from scratch example
- [x] Eager mode training with `tf.GradientTape`
- [x] Graph mode training with `model.fit`
- [x] Functional model with `tf.keras.layers`
- [x] Input pipeline using `tf.data`
- [x] Vectorized transformations
- [x] GPU accelerated
- [x] Fully integrated with `absl-py` abseil.io
- [x] Clean implementation
- [x] Following the best practices
- [x] MIT License

![demo](https://raw.githubusercontent.com/zzh8829/yolov3-tf2/master/data/meme_out.jpg)

## Usage

### Installation

```
pip install -r requirements.txt
```

### Convert pre-trained Darknet weights

```bash
# yolov3
wget https://pjreddie.com/media/files/yolov3.weights -O data/yolov3.weights
python convert.py

# yolov3-tiny
wget https://pjreddie.com/media/files/yolov3-tiny.weights -O data/yolov3-tiny.weights
python convert.py --weights ./data/yolov3-tiny.weights --output ./data/yolov3-tiny.h5 --tiny
```

### Detection

```bash
# yolov3
python detect.py

# yolov3-tiny
python detect.py --weights ./data/yolov3-tiny.h5 --tiny
```

### Training (WIP)

``` bash
python train.py
```

## Implementation Details

### Eager execution

Great addition for existing Tensorflow experts.
Not very easy to use without some intermediate understanding of Tensorflow graphs.

### GradientTape

Extremely useful for debugging purpose, you can set breakpoints anywhere.
Downside is you have to re-implementing all the model.fit features

### @tf.function

@tf.function is very cool. Do have some caveats tho.

### Loading pre-trained Darknet weights

very hard with pure functional API because the layer ordering is different in
tf.keras and darknet. The clean solution here is creating sub-models in keras.

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
