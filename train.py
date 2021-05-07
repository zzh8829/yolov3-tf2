import tensorflow as tf
import os

from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from yolov3_tf2.models import YoloV3, YoloLoss, yolo_anchors, yolo_anchor_masks
from yolov3_tf2.utils import freeze_all
import yolov3_tf2.dataset as dataset
import valohai


params = {
    "batch_size": 8,
    "epochs": 10,
    "learning_rate": 0.001,
    "weights_num_classes": 80,
    "size": 416,
}

inputs = {
    "classes": ".data/classes.txt",
    "train": ".data/train/train.tfrecord",
    "test": ".data/test/test.tfrecord",
    "model": ".data/weights/yolov3.tf.data-00000-of-00001",
}

valohai.prepare(step="train", default_parameters=params, default_inputs=inputs)


class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        with valohai.logger() as epoch_logger:
            keys = list(logs.keys())
            epoch_logger.log("epoch", epoch)
            for key in keys:
                epoch_logger.log(key, logs[key])


def main():
    with open(valohai.inputs('classes').path(), "r") as f:
        num_classes = len([line.strip("\n") for line in f if line != "\n"])
        print(f"Number of classes used: {num_classes}")

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    model = YoloV3(
        valohai.parameters('size').value,
        training=True,
        classes=num_classes
    )
    anchors = yolo_anchors
    anchor_masks = yolo_anchor_masks

    train_dataset = dataset.load_tfrecord_dataset(
        valohai.inputs('train').path(),
        valohai.inputs('classes').path(),
        valohai.parameters('size').value,
    )
    train_dataset = train_dataset.shuffle(buffer_size=512)
    train_dataset = train_dataset.batch(valohai.parameters('batch_size').value)
    train_dataset = train_dataset.map(lambda x, y: (
        dataset.transform_images(x, valohai.parameters('size').value),
        dataset.transform_targets(y, anchors, anchor_masks, valohai.parameters('size').value)
    ))
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    test_dataset = dataset.load_tfrecord_dataset(
        valohai.inputs('test').path(),
        valohai.inputs('classes').path(),
        valohai.parameters('size').value,
    )
    test_dataset = test_dataset.batch(valohai.parameters('batch_size').value)
    test_dataset = test_dataset.map(lambda x, y: (
        dataset.transform_images(x, valohai.parameters('size').value),
        dataset.transform_targets(y, anchors, anchor_masks, valohai.parameters('size').value)
    ))

    model_pretrained = YoloV3(
        valohai.parameters('size').value,
        training=True,
        classes=valohai.parameters('weights_num_classes').value or num_classes
    )
    weights_path = os.path.join(os.path.dirname(valohai.inputs('model').path()), "model.tf")
    model_pretrained.load_weights(weights_path).expect_partial()
    model.get_layer('yolo_darknet').set_weights(model_pretrained.get_layer('yolo_darknet').get_weights())
    freeze_all(model.get_layer('yolo_darknet'))

    optimizer = tf.keras.optimizers.Adam(lr=valohai.parameters('learning_rate').value)
    loss = [YoloLoss(anchors[mask], classes=num_classes) for mask in anchor_masks]

    model.compile(optimizer=optimizer, loss=loss, run_eagerly=False)

    callbacks = [
        ReduceLROnPlateau(verbose=1),
        ModelCheckpoint(
            valohai.outputs('checkpoints').path('model.tf'),
            verbose=1,
            save_weights_only=True
        ),
        CustomCallback(),
    ]

    model.fit(train_dataset,
              epochs=valohai.parameters('epochs').value,
              callbacks=callbacks,
              validation_data=test_dataset)
    path = valohai.outputs('model').path('model.tf')
    model.save_weights(valohai.outputs('model').path('model.tf'))
    print(f'Model saved to {path}')


if __name__ == "__main__":
    main()
