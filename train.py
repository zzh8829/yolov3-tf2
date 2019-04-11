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
    YoloV3, YoloLoss, yolo_output, yolo_anchors, yolo_anchor_masks
)
import yolov3_tf2.dataset as dataset


flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './data/yolov3.h5', 'path to weights file')
flags.DEFINE_string('dataset', '', 'path to dataset')
flags.DEFINE_enum('mode', 'test', ['scratch', 'transfer', 'test'],
                  'Training mode')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_boolean('eager', False, 'train eagerly with gradient tape')
flags.DEFINE_integer('size', 416, 'image size')
flags.DEFINE_integer('epochs', 2, 'number of epochs')
flags.DEFINE_integer('batch_size', 8, 'batch size')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')


def main(_argv):
    FLAGS.eager = True
    # train_dataset = dataset.load_tfrecord_dataset(
    #     FLAGS.dataset, FLAGS.classes)
    train_dataset = dataset.load_fake_dataset()
    train_dataset = train_dataset.shuffle(buffer_size=1024)  # TODO: not 1024
    train_dataset = train_dataset.batch(FLAGS.batch_size)
    train_dataset = train_dataset.map(lambda x, y: (
        dataset.transform_images(x, FLAGS.size),
        dataset.transform_targets(y, yolo_anchors, yolo_anchor_masks, 80)))
    train_dataset = train_dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)

    val_dataset = dataset.load_fake_dataset()
    val_dataset = val_dataset.batch(FLAGS.batch_size)
    val_dataset = val_dataset.map(lambda x, y: (
        dataset.transform_images(x, FLAGS.size),
        dataset.transform_targets(y, yolo_anchors, yolo_anchor_masks, 80)))

    model = YoloV3(FLAGS.size)
    init_weights = [l.get_weights() for l in model.layers]
    model.load_weights(FLAGS.weights)

    # for l in model.layers[:185]:  # darknet-53 layers TODO: refactor
    #     l.trainable = False
    # for i in range(185, len(model.layers)):
    #     model.layers[i].set_weights(init_weights[i])
    # for l in model.layers:
    #     if l.name.startswith('batch_norm'):
    #         l.trainable = False
    for l in model.layers:
        l.trainable = False
    model.layers[240].trainable = True  # conv2d before output
    model.layers[249].trainable = True  # conv2d output_1

    optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate)
    loss = [YoloLoss(yolo_anchors[mask]) for mask in yolo_anchor_masks]

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
