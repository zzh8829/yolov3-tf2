# Training Instruction

## VOC 2012 Dataset from Scratch

Full instruction on how to train using VOC 2012 from scratch

Requirement:
  1. Able to detect image using pretrained darknet model
  2. Many Gigabytes of Disk Space
  3. High Speed Internet Connection Preferred
  4. GPU Preferred


### 1. Download Dataset

You can read the full description of dataset [here](http://host.robots.ox.ac.uk/pascal/VOC/)
```bash
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar -O ./data/voc2012_raw.tar
mkdir -p ./data/voc2012_raw
tar -xf ./data/voc2012_raw.tar -C ./data/voc2012_raw
ls ./data/voc2012_raw/VOCdevkit/VOC2012 # Explore the dataset
```

### 2. Transform Dataset

See tools/voc2012.py for implementation, this format is based on [tensorflow object detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). Many fields 
are not required, I left them there for compatibility with official API.

```bash
python tools/voc2012.py \
  --data_dir './data/voc2012_raw/VOCdevkit/VOC2012'
	--split train \
	--output_file ./data/voc2012_train.tfrecord 

python tools/voc2012.py \
  --data_dir './data/voc2012_raw/VOCdevkit/VOC2012'
	--split val \
	--output_file ./data/voc2012_val.tfrecord 
```

### 3. Training

You can adjust the parameters based on your setup

```bash
python train.py \
	--dataset ./data/voc2012_train.tfrecord \
	--val_dataset ./data/voc2012_val.tfrecord \
	--classes ./data/voc2012.names \
	--num_classes 20 \
	--mode fit --transfer none \
	--batch_size 16 \
	--epochs 3 \
```

I have tested this works 100% with correct loss and converging over time
Each epoch takes around 10 minutes on single AWS p2.xlarge (Nvidia K80 GPU) Instance.

### 4. Inference

```bash
python detect.py \
	--classes ./data/voc2012.names \
	--num_classes 20 \
	--weights ./checkpoints/yolov3_train_3.tf
```

You should see some detect objects in the standard output and the visualization at `output.jpg`.
this is just a proof of concept, so it won't be as good as pretrained models

