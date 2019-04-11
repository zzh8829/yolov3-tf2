from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny, YoloLoss, yolo_output,
    yolo_anchors, yolo_anchor_masks,
    yolo_tiny_anchors, yolo_tiny_anchor_masks
)
from yolov3_tf2.batch_norm import freeze_bn
import yolov3_tf2.dataset as dataset


flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './data/yolov3.h5', 'path to weights file')
flags.DEFINE_string('dataset', '', 'path to dataset')
flags.DEFINE_string('val_dataset', '', 'path to validation dataset')
flags.DEFINE_enum('mode', 'transfer_last',
                  ['scratch', 'transfer', 'transfer_last', 'frozen'],
                  'Training mode')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_boolean('eager', True, 'train eagerly with gradient tape')
flags.DEFINE_integer('size', 416, 'image size')
flags.DEFINE_integer('epochs', 2, 'number of epochs')
flags.DEFINE_integer('batch_size', 8, 'batch size')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')


def main(_argv):
    if FLAGS.tiny:
        model = YoloV3Tiny(FLAGS.size)
        anchors = yolo_tiny_anchors
        anchor_masks = yolo_tiny_anchor_masks
    else:
        model = YoloV3(FLAGS.size)
        anchors = yolo_anchors
        anchor_masks = yolo_anchor_masks

    train_dataset = dataset.load_tfrecord_dataset(
        FLAGS.dataset, FLAGS.classes)
    train_dataset = train_dataset.shuffle(buffer_size=1024)  # TODO: not 1024
    train_dataset = train_dataset.batch(FLAGS.batch_size)
    train_dataset = train_dataset.map(lambda x, y: (
        dataset.transform_images(x, FLAGS.size),
        dataset.transform_targets(y, anchors, anchor_masks, 80)))
    train_dataset = train_dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)

    # val_dataset = dataset.load_fake_dataset()
    val_dataset = dataset.load_tfrecord_dataset(
        FLAGS.val_dataset, FLAGS.classes)
    val_dataset = val_dataset.batch(FLAGS.batch_size)
    val_dataset = val_dataset.map(lambda x, y: (
        dataset.transform_images(x, FLAGS.size),
        dataset.transform_targets(y, anchors, anchor_masks, 80)))

    if FLAGS.mode == 'scratch':
        pass  # training from scratch (not recommended)
    else:  # transfer learning mode
        model.get_layer('yolo_body').load_weights(FLAGS.weights)
        for l in model.get_layer('yolo_body').get_layer('yolo_darknet').layers:
            l.trainable = False

        if FLAGS.tiny:  # get initial weights
            init_model = YoloV3Tiny(FLAGS.size)
        else:
            init_model = YoloV3(FLAGS.size)

        if FLAGS.mode == 'transfer':  # learn all non darknet layers
            for l in model.get_layer('yolo_body').layers:
                if l.name != 'yolo_darknet' and l.name.startswith('yolo_'):
                    l.set_weights(init_model.get_layer('yolo_body').get_layer(
                        l.name).get_weights())
        elif FLAGS.mode == 'transfer_last':  # only learn output layer
            for l in model.get_layer('yolo_body').layers:
                if 'yolo_output' in l.name:
                    l.set_weights(init_model.get_layer('yolo_body').get_layer(
                        l.name).get_weights())
                    freeze_bn(l, False)
                else:
                    l.trainable = False
                    freeze_bn(l, True)
        elif FLAGS.mode == 'frozen':  # learn nothing
            freeze_bn(model, True)
            for l in model.get_layer('yolo_body').layers:
                l.trainable = False

    model = model.get_layer('yolo_body')

    optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate)
    loss = [YoloLoss(anchors[mask]) for mask in anchor_masks]

    if FLAGS.eager:
        avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)

        for epoch in range(FLAGS.epochs):
            for batch, (images, labels) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    outputs = model(images, training=True)
                    regularization_loss = tf.math.add_n(model.losses)
                    pred_loss = tf.zeros(tf.shape(images)[0])
                    for output, label, loss_fn in zip(outputs, labels, loss):
                        pred_loss += loss_fn(label, output)
                    total_loss = pred_loss + regularization_loss

                grads = tape.gradient(total_loss, model.trainable_variables)
                optimizer.apply_gradients(
                    zip(grads, model.trainable_variables))

                logging.info("{}_{}, {}".format(
                    epoch, batch, total_loss.numpy()))
                avg_loss.update_state(total_loss)

            logging.info("{}, {}".format(epoch, avg_loss.result().numpy()))
            avg_loss.reset_states()
            model.save('checkpoints/yolov3_train_{}.h5'.format(epoch))
    else:
        model.compile(optimizer=optimizer, loss=loss)

        callbacks = [
            ReduceLROnPlateau(),
            EarlyStopping(),
            ModelCheckpoint('checkpoints/yolov3_train.h5',
                            save_best_only=True, save_weights_only=True),
            TensorBoard(log_dir='logs')
        ]

        history = model.fit(train_dataset,
                            epochs=FLAGS.epochs,
                            callbacks=callbacks,
                            validation_data=val_dataset)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
