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
    YoloV3, YoloV3Tiny, YoloLoss,
    yolo_anchors, yolo_anchor_masks,
    yolo_tiny_anchors, yolo_tiny_anchor_masks
)
from yolov3_tf2.utils import freeze_all
import yolov3_tf2.dataset as dataset

flags.DEFINE_string('dataset', '', 'path to dataset')
flags.DEFINE_string('val_dataset', '', 'path to validation dataset')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_enum('mode', 'fit', ['fit', 'eager_fit', 'eager_tf'],
                  'fit: model.fit, '
                  'eager_fit: model.fit(run_eagerly=True), '
                  'eager_tf: custom GradientTape')
flags.DEFINE_enum('transfer', 'none',
                  ['none', 'yolo_darknet', 'yolo_conv', 'yolo_output_conv', 'all'],
                  'none: Training from scratch (no weights transfer), '
                  'yolo_darknet: Transfer darknet sub-model weights, '
                  'yolo_conv: Transfer darknet and conv sub-model weights, '
                  'yolo_output_conv: Transfer darknet and conv sub-model weights and first output conv layer weights, '
                  'all: Transfer all weights (pretrained weights need to have the same number of classes)')
flags.DEFINE_enum('freeze', 'none',
                  ['none', 'yolo_darknet', 'yolo_conv', 'yolo_output_conv', 'all'],
                  'none: Tune all weights, '
                  'yolo_darknet: Tune all but darknet sub-model weights, '
                  'yolo_conv: Tune output sub-model weights, '
                  'yolo_output_conv: Tune only output sub-model without the first conv layer, '
                  'all: Do not allow tuning of weights')
flags.DEFINE_integer('size', 416, 'image size')
flags.DEFINE_integer('epochs', 2, 'number of epochs')
flags.DEFINE_integer('batch_size', 8, 'batch size')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
flags.DEFINE_integer('weights_num_classes', None, 'specify num class for `weights` file if different, '
                     'useful in transfer learning with different number of classes')


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if FLAGS.tiny:
        model = YoloV3Tiny(FLAGS.size, training=True,
                           classes=FLAGS.num_classes)
        anchors = yolo_tiny_anchors
        anchor_masks = yolo_tiny_anchor_masks
    else:
        model = YoloV3(FLAGS.size, training=True, classes=FLAGS.num_classes)
        anchors = yolo_anchors
        anchor_masks = yolo_anchor_masks

    train_dataset = dataset.load_fake_dataset()
    if FLAGS.dataset:
        train_dataset = dataset.load_tfrecord_dataset(
            FLAGS.dataset, FLAGS.classes, FLAGS.size)
    train_dataset = train_dataset.shuffle(buffer_size=512)
    train_dataset = train_dataset.batch(FLAGS.batch_size)
    train_dataset = train_dataset.map(lambda x, y: (
        dataset.transform_images(x, FLAGS.size),
        dataset.transform_targets(y, anchors, anchor_masks, FLAGS.size)))
    train_dataset = train_dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)

    val_dataset = dataset.load_fake_dataset()
    if FLAGS.val_dataset:
        val_dataset = dataset.load_tfrecord_dataset(
            FLAGS.val_dataset, FLAGS.classes, FLAGS.size)
    val_dataset = val_dataset.batch(FLAGS.batch_size)
    val_dataset = val_dataset.map(lambda x, y: (
        dataset.transform_images(x, FLAGS.size),
        dataset.transform_targets(y, anchors, anchor_masks, FLAGS.size)))

    # Configure the model for transfer learning
    if FLAGS.transfer != 'none':
        # if we need all weights, no need to create another model
        if FLAGS.transfer == 'all':
            model.load_weights(FLAGS.weights)

        # else, we need only some of the weights
        # create appropriate model_pretrained, load all weights and copy the ones we need
        else:
            if FLAGS.tiny:
                model_pretrained = YoloV3Tiny(FLAGS.size, training=True, classes=FLAGS.weights_num_classes or FLAGS.num_classes)
            else:
                model_pretrained = YoloV3(FLAGS.size, training=True, classes=FLAGS.weights_num_classes or FLAGS.num_classes)
            # load pretrained weights
            model_pretrained.load_weights(FLAGS.weights)
            # transfer darknet
            darknet = model.get_layer('yolo_darknet')
            darknet.set_weights(model_pretrained.get_layer('yolo_darknet').get_weights())
            # transfer 'yolo_conv_i' layer weights
            if FLAGS.transfer in ['yolo_conv', 'yolo_output_conv']:
                for l in model.layers:
                   if l.name.startswith('yolo_conv'):
                       model.get_layer(l.name).set_weights(model_pretrained.get_layer(l.name).get_weights())
            # transfer 'yolo_output_i' first conv2d layer
            if FLAGS.transfer == 'yolo_output_conv':
                # transfer tiny output conv2d
                if FLAGS.tiny:
                    # get and set the weights of the appropriate layers
                    model.layers[4].layers[1].set_weights(model_pretrained.layers[4].layers[1].get_weights())
                    model.layers[5].layers[1].set_weights(model_pretrained.layers[5].layers[1].get_weights())
                    # should I freeze batch_norm as well?
                else:
                    # get and set the weights of the appropriate layers
                    model.layers[5].layers[1].set_weights(model_pretrained.layers[5].layers[1].get_weights())
                    model.layers[6].layers[1].set_weights(model_pretrained.layers[6].layers[1].get_weights())
                    model.layers[7].layers[1].set_weights(model_pretrained.layers[7].layers[1].get_weights())
                    # should I freeze batch_norm as well?
    # no transfer learning
    else:
        pass

    # freeze layers, if requested
    if FLAGS.freeze != 'none':
        if FLAGS.freeze == 'all':
            freeze_all(model)
        if FLAGS.freeze in ['yolo_darknet' 'yolo_conv', 'yolo_output_conv']:
            darknet = model.get_layer('yolo_darknet')
            freeze_all(darknet)
        if FLAGS.freeze in ['yolo_conv', 'yolo_output_conv']:
            for l in model.layers:
                if l.name.startswith('yolo_conv'):
                    freeze_all(l)
        if FLAGS.freeze == 'yolo_output_conv':
            if FLAGS.tiny:
                # freeze the appropriate layers
                freeze_all(model.layers[4].layers[1])
                freeze_all(model.layers[5].layers[1])
            else:
                # freeze the appropriate layers
                freeze_all(model.layers[5].layers[1])
                freeze_all(model.layers[6].layers[1])
                freeze_all(model.layers[7].layers[1])
    # freeze nothing
    else:
        pass
    optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate)
    loss = [YoloLoss(anchors[mask], classes=FLAGS.num_classes)
            for mask in anchor_masks]

    if FLAGS.mode == 'eager_tf':
        # Eager mode is great for debugging
        # Non eager graph mode is recommended for real training
        avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
        avg_val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)

        for epoch in range(1, FLAGS.epochs + 1):
            for batch, (images, labels) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    outputs = model(images, training=True)
                    regularization_loss = tf.reduce_sum(model.losses)
                    pred_loss = []
                    for output, label, loss_fn in zip(outputs, labels, loss):
                        pred_loss.append(loss_fn(label, output))
                    total_loss = tf.reduce_sum(pred_loss) + regularization_loss

                grads = tape.gradient(total_loss, model.trainable_variables)
                optimizer.apply_gradients(
                    zip(grads, model.trainable_variables))

                logging.info("{}_train_{}, {}, {}".format(
                    epoch, batch, total_loss.numpy(),
                    list(map(lambda x: np.sum(x.numpy()), pred_loss))))
                avg_loss.update_state(total_loss)

            for batch, (images, labels) in enumerate(val_dataset):
                outputs = model(images)
                regularization_loss = tf.reduce_sum(model.losses)
                pred_loss = []
                for output, label, loss_fn in zip(outputs, labels, loss):
                    pred_loss.append(loss_fn(label, output))
                total_loss = tf.reduce_sum(pred_loss) + regularization_loss

                logging.info("{}_val_{}, {}, {}".format(
                    epoch, batch, total_loss.numpy(),
                    list(map(lambda x: np.sum(x.numpy()), pred_loss))))
                avg_val_loss.update_state(total_loss)

            logging.info("{}, train: {}, val: {}".format(
                epoch,
                avg_loss.result().numpy(),
                avg_val_loss.result().numpy()))

            avg_loss.reset_states()
            avg_val_loss.reset_states()
            model.save_weights(
                'checkpoints/yolov3_train_{}.tf'.format(epoch))
    else:
        model.compile(optimizer=optimizer, loss=loss,
                      run_eagerly=(FLAGS.mode == 'eager_fit'))

        callbacks = [
            ReduceLROnPlateau(verbose=1),
            EarlyStopping(patience=3, verbose=1),
            ModelCheckpoint('checkpoints/yolov3_train_{epoch}.tf',
                            verbose=1, save_weights_only=True),
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
