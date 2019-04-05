from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
import numpy as np
import cv2

from yolov3_tf2.models import YoloV3, YoloLoss, yolo_anchors, yolo_output
from yolov3_tf2.utils import draw_outputs


flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './data/yolov3.h5', 'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_string('image', './data/girl.png', 'path to input image')
flags.DEFINE_string('output', './data/output.jpg', 'path to output image')


@tf.function
def transform_targets(y_true, grid_size, anchor_idxs, classes):
    # y_true: (N, boxes, (x1, y1, x2, y2, class, best_anchor))
    N = tf.shape(y_true)[0]

    # y_true_out: (N, grid, grid, anchors, [x, y, w, h, obj, class])
    y_true_out = tf.zeros(
        (N, grid_size, grid_size, tf.shape(anchor_idxs)[0], 6))

    anchor_idxs = tf.cast(anchor_idxs, tf.int32)

    indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
    updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
    idx = 0
    for i in tf.range(N):
        for j in tf.range(tf.shape(y_true)[1]):
            if tf.equal(y_true[i][j][2], 0):
                continue
            anchor_eq = tf.equal(
                anchor_idxs, tf.cast(y_true[i][j][5], tf.int32))

            if tf.reduce_any(anchor_eq):
                box = y_true[i][j][0:4]
                box_xy = (y_true[i][j][0:2] + y_true[i][j][2:4]) / 2

                anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)
                grid_xy = tf.cast(box_xy // (1/grid_size), tf.int32)

                # grid[y][x][anchor] = (tx, ty, bw, bh, obj, class)
                indexes = indexes.write(
                    idx, [i, grid_xy[1], grid_xy[0], anchor_idx[0][0]])
                updates = updates.write(
                    idx, [box[0], box[1], box[2], box[3], 1, y_true[i][j][4]])
                idx += 1

    # tf.print(indexes.stack())
    # tf.print(updates.stack())

    return tf.tensor_scatter_nd_update(
        y_true_out, indexes.stack(), updates.stack())


def transform_fn(x_train, y_train):
    y_outs = []
    grid_size = 13

    # calculate anchor index for true boxes
    anchors = tf.cast(yolo_anchors, tf.float32)
    anchor_area = anchors[..., 0] * anchors[..., 1]
    box_wh = y_train[..., 2:4] - y_train[..., 0:2]
    box_wh = tf.tile(tf.expand_dims(box_wh, -2),
                     (1, 1, tf.shape(anchors)[0], 1))
    box_area = box_wh[..., 0] * box_wh[..., 1]
    intersection = tf.minimum(box_wh[..., 0], anchors[..., 0]) * \
        tf.minimum(box_wh[..., 1], anchors[..., 1])
    iou = intersection / (box_area + anchor_area - intersection)
    anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.float32)
    anchor_idx = tf.expand_dims(anchor_idx, axis=-1)

    y_train = tf.concat([y_train, anchor_idx], axis=-1)

    for anchor_idxs in [[6, 7, 8], [3, 4, 5], [0, 1, 2]]:
        y_outs.append(transform_targets(y_train, grid_size, anchor_idxs, 80))
        grid_size *= 2

    return x_train, tuple(y_outs)


def main(_argv):
    N = 200

    img = img_og = cv2.imread('./data/girl.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (416, 416))
    img = img / 255.0
    x_train = np.expand_dims(img, 0)
    # [batch, width, height, channel]
    x_train = np.tile(x_train, (N, 1, 1, 1)).astype(np.float32)

    labels = [
        [0.18494931, 0.03049111, 0.9435849,  0.96302897, 0],
        [0.01586703, 0.35938117, 0.17582396, 0.6069674, 56],
        [0.09158827, 0.48252046, 0.26967454, 0.6403017, 67]
    ] + [[0, 0, 0, 0, 0]] * 17
    y_train = np.expand_dims(np.array(labels), 0)
    # [batch, boxes, [x1, y1, x2, y2, class]]
    y_train = np.tile(y_train, (N, 1, 1)).astype(np.float32)

    logging.info("{}, {}".format(x_train.shape, y_train.shape))

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024)
    train_dataset = train_dataset.batch(10)
    train_dataset = train_dataset.map(transform_fn)
    train_dataset = train_dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)

    model = YoloV3(416)
    model.load_weights('./data/yolov3.h5')

    optimizer = tf.keras.optimizers.Adam()
    loss = [YoloLoss(yolo_anchors[6:9]),
            YoloLoss(yolo_anchors[3:6]),
            YoloLoss(yolo_anchors[0:3])]

    avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)

    for l in model.layers[:185]:  # darknet-53 layers TODO: refactor
        l.trainable = False
    for l in model.layers:
        if l.name.startswith('batch_norm'):
            l.trainable = False

    for batch, (images, labels) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            outputs = model(images, training=True)

            loss_value = tf.zeros(tf.shape(images)[0])
            for output, label, loss_fn in zip(outputs, labels, loss):
                loss_value += loss_fn(label, output)

            grads = tape.gradient(loss_value, model.trainable_variables)
        avg_loss(loss_value)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        logging.info("{}, {}".format(batch, avg_loss.result().numpy()))

    model.save('data/yolov3_custom2.h5')


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
