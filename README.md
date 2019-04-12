# YoloV3 Implemented in TensorFlow 2.0

This repo provides a clean implementation of YoloV3 in TensorFlow 2.0 using all the best practices.

## Key Features

- [x] TensorFlow 2.0
- [x] `yolov3` with pre-trained Weights
- [x] `yolov3-tiny` with pre-trained Weights
- [x] Inference example
- [x] Transfer learning example
- [x] Eager mode training with `tf.GradientTape`
- [x] Graph mode training with `model.fit`
- [x] Functional model with `tf.keras.layers`
- [x] Input pipeline using `tf.data`
- [x] Vectorized transformations
- [x] GPU accelerated
- [x] Fully integrated with `absl-py` from [abseil.io](https://abseil.io)
- [x] Clean implementation
- [x] Following the best practices
- [x] MIT License

![demo](https://raw.githubusercontent.com/zzh8829/yolov3-tf2/master/data/meme_out.jpg)
![demo](https://raw.githubusercontent.com/zzh8829/yolov3-tf2/master/data/street_out.jpg)

## Usage

### Installation

```bash
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
python detect.py --image ./data/street.jpg

# yolov3-tiny
python detect.py --weights ./data/yolov3-tiny.h5 --tiny
```

### Training

You need to generate tfrecord following the TensorFlow Object Detection API
For example you can use [Microsoft VOTT](https://github.com/Microsoft/VoTT) to generate such dataset
You can also use this [script](https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_pascal_tf_record.py) to create pascal voc dataset.


``` bash
python train.py --batch_size 8 --dataset ~/Data/voc2012.tfrecord --val_dataset ~/Data/voc2012_val.tfrecord --epochs 100 --noeager --mode transfer

python train.py --batch_size 8 --dataset ~/Data/voc2012.tfrecord --val_dataset ~/Data/voc2012_val.tfrecord --epochs 100 --noeager --mode transfer_last

python train.py --batch_size 8 --dataset ~/Data/voc2012.tfrecord --val_dataset ~/Data/voc2012_val.tfrecord --epochs 100 --noeager --mode scratch

python train.py --batch_size 8 --dataset ~/Data/voc2012.tfrecord --val_dataset ~/Data/voc2012_val.tfrecord --epochs 100 --noeager --mode transfer --weights ./data/yolov3-tiny.h5 --tiny
```

## Command Line Args

```bash
convert.py:
  --output: path to output
    (default: './data/yolov3.h5')
  --[no]tiny: yolov3 or yolov3-tiny
    (default: 'false')
  --weights: path to weights file
    (default: './data/yolov3.weights')

detect.py:
  --classes: path to classes file
    (default: './data/coco.names')
  --image: path to input image
    (default: './data/girl.png')
  --output: path to output image
    (default: './data/output.jpg')
  --[no]tiny: yolov3 or yolov3-tiny
    (default: 'false')
  --weights: path to weights file
    (default: './data/yolov3.h5')

train.py:
  --batch_size: batch size
    (default: '8')
    (an integer)
  --classes: path to classes file
    (default: './data/coco.names')
  --dataset: path to dataset
    (default: '')
  --[no]eager: train eagerly with gradient tape
    (default: 'true')
  --epochs: number of epochs
    (default: '2')
    (an integer)
  --learning_rate: learning rate
    (default: '0.001')
    (a number)
  --mode: <scratch|transfer|transfer_last|frozen>: Training mode
    (default: 'transfer_last')
  --size: image size
    (default: '416')
    (an integer)
  --[no]tiny: yolov3 or yolov3-tiny
    (default: 'false')
  --val_dataset: path to validation dataset
    (default: '')
  --weights: path to weights file
    (default: './data/yolov3.h5')
```

## Implementation Details

### Eager execution

Great addition for existing TensorFlow experts.
Not very easy to use without some intermediate understanding of TensorFlow graphs.

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
