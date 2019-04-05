import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Add,
    Concatenate,
    Conv2D,
    Input,
    LeakyReLU,
    MaxPool2D,
    UpSampling2D,
    ZeroPadding2D,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import (
    binary_crossentropy,
    sparse_categorical_crossentropy
)
from tensorflow.python.keras.backend import get_graph  # extreme hack
from .utils import broadcast_iou


yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)]) / 416.0

yolo_tiny_anchors = np.array([(10, 14), (23, 27), (37, 58),
                              (81, 82), (135, 169),  (344, 319)]) / 416.0


class BatchNormalization(tf.keras.layers.BatchNormalization):
    """
    Make trainable=False freeze BN for real (the og version is sad)
    """

    def call(self, x, training=False):
        if training is None:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)


def DarknetConv(x, filters, size, strides=1, batch_norm=True):
    if strides == 1:
        padding = 'same'
    else:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)  # top left half-padding
        padding = 'valid'
    x = Conv2D(filters=filters, kernel_size=size,
               strides=strides, padding=padding,
               use_bias=not batch_norm, kernel_regularizer=l2(0.9))(x)
    if batch_norm:
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
    return x


def DarknetResidual(x, filters):
    prev = x
    x = DarknetConv(x, filters // 2, 1)
    x = DarknetConv(x, filters, 3)
    x = Add()([prev, x])
    return x


def DarknetBlock(x, filters, blocks):
    x = DarknetConv(x, filters, 3, strides=2)
    for _ in range(blocks):
        x = DarknetResidual(x, filters)
    return x


def Darknet53(x):
    x = DarknetConv(x, 32, 3)
    x = DarknetBlock(x, 64, 1)
    x = DarknetBlock(x, 128, 2)  # skip connection
    x = x_36 = DarknetBlock(x, 256, 8)  # skip connection
    x = x_61 = DarknetBlock(x, 512, 8)
    x = DarknetBlock(x, 1024, 4)
    return x_36, x_61, x


def Darknet(x):
    x = DarknetConv(x, 16, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 32, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 64, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 128, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = x_8 = DarknetConv(x, 256, 3)  # skip connection
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 512, 3)
    x = MaxPool2D(2, 1, 'same')(x)
    x = DarknetConv(x, 1024, 3)
    return x_8, x


def YoloConcat(x, x_concat, filters):
    x = DarknetConv(x, filters, 1)
    x = UpSampling2D(2)(x)
    x = Concatenate()([x, x_concat])
    return x


def YoloOutput(x, filters, out_filters):
    x = DarknetConv(x, filters, 1)
    x = DarknetConv(x, filters * 2, 3)
    x = DarknetConv(x, filters, 1)
    x = DarknetConv(x, filters * 2, 3)
    x = x_out = DarknetConv(x, filters, 1)
    x_out = DarknetConv(x_out, filters * 2, 3)
    x_out = DarknetConv(x_out, out_filters, 1, batch_norm=False)
    return x, x_out


def YoloV3(size=None, classes=80):
    with get_graph().as_default():
        x = inputs = Input(shape=[size, size, 3])

        with tf.name_scope("darknet"):
            x_36, x_61, x = Darknet53(x)

        with tf.name_scope("yolo"):
            x, output_1 = YoloOutput(x, 512, 3*(classes+5))

        with tf.name_scope("yolo"):
            x = YoloConcat(x, x_61, 256)
            x, output_2 = YoloOutput(x, 256, 3*(classes+5))

        with tf.name_scope("yolo"):
            x = YoloConcat(x, x_36, 128)
            x, output_3 = YoloOutput(x, 128, 3*(classes+5))

    return Model(inputs, [output_1, output_2, output_3])


def YoloTinyOutput(x, filters, out_filters):
    x = DarknetConv(x, filters, 3)
    x = DarknetConv(x, out_filters, 1, batch_norm=False)
    return x


def YoloV3Tiny(size=None, classes=80):
    x = inputs = Input(shape=[size, size, 3])
    x_8, x = Darknet(x)

    x = DarknetConv(x, 256, 1)
    output_1 = YoloTinyOutput(x, 512, 3*(classes+5))

    x = YoloConcat(x, x_8, 128)
    output_2 = YoloTinyOutput(x, 256, 3*(classes+5))

    return Model(inputs, [output_1, output_2])


def yolo_boxes(pred, anchors, classes=80):
    # [batch_size, size, size, anchors,
    #     [center_x, center_y, width, height, confidence, ...classes]]
    grid_size = pred.shape[1]
    pred = tf.reshape(pred, (-1, grid_size, grid_size,
                             tf.shape(anchors)[0], classes + 5))
    box_xy, box_wh, objectness, class_probs = tf.split(
        pred, (2, 2, 1, classes), axis=-1)

    box_xy = tf.sigmoid(box_xy)
    objectness = tf.sigmoid(objectness)
    class_probs = tf.sigmoid(class_probs)

    # !!! grid[x][y] == (y, x)
    grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    grid = tf.expand_dims(tf.stack(grid, axis=-1),
                          axis=2)  # [size, size, 1, 2]
    grid = tf.tile(grid, (1, 1, tf.shape(anchors)[0], 1))

    box_xy = (box_xy + tf.cast(grid, tf.float32)) / grid_size
    box_wh = tf.exp(box_wh) * anchors

    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    box = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return box, objectness, class_probs


def yolo_loss(y_true, y_pred, anchors, classes, ignore_thresh):
    grid_size = y_pred.shape[1]

    # 1. transform all pred outputs
    pred_box, _, _ = yolo_boxes(y_pred, anchors)
    # (N, gridx, gridy, anchors, (x, y, w, h, obj, ...class))
    y_pred = tf.reshape(
        y_pred, (-1, grid_size, grid_size, len(anchors), classes + 5))
    pred_xy, pred_wh, pred_obj, pred_class = tf.split(
        y_pred, (2, 2, 1, classes), axis=-1)
    pred_xy = tf.sigmoid(pred_xy)

    # 2. transform all true outputs
    true_box, true_obj, true_class_idx = tf.split(y_true, (4, 1, 1), axis=-1)
    true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
    true_wh = true_box[..., 2:4] - true_box[..., 0:2]

    # give higher weights to small boxes
    box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]

    # 3. inverting the pred box equations
    true_xy = (true_xy % (1/grid_size)) / (1/grid_size)
    true_wh = tf.math.log(true_wh / anchors)
    true_wh = tf.where(tf.math.is_inf(true_wh),
                       tf.zeros_like(true_wh), true_wh)

    # 4. calculate all masks
    obj_mask = tf.squeeze(true_obj, -1)
    # ignore false positive when iou is over threshold
    true_box_flat = tf.boolean_mask(true_box, tf.cast(obj_mask, tf.bool))
    best_iou = tf.reduce_max(broadcast_iou(pred_box, true_box_flat), axis=-1)
    ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)

    # 5. calculate all losses
    xy_loss = obj_mask * box_loss_scale * \
        tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
    wh_loss = obj_mask * box_loss_scale * \
        tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
    obj_loss = binary_crossentropy(true_obj, pred_obj, from_logits=True)
    obj_loss = obj_mask * obj_loss + (1 - obj_mask) * ignore_mask * obj_loss
    # TODO: use binary_crossentropy instead
    class_loss = obj_mask * sparse_categorical_crossentropy(
        true_class_idx, pred_class, from_logits=True)

    # 6. sum over (gridx, gridy, anchors) => (batch, )
    xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
    wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
    obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
    class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))

    return xy_loss, wh_loss, obj_loss, class_loss


def YoloLoss(anchors, classes=80, ignore_thresh=0.5):
    def loss(y_true, y_pred):
        xy_loss, wh_loss, obj_loss, class_loss = yolo_loss(
            y_true, y_pred, anchors, classes, ignore_thresh)
        return xy_loss + wh_loss + obj_loss + class_loss
    return loss


def yolo_output(outputs):
    b, s, c = [], [], []

    o1 = yolo_boxes(outputs[0], yolo_anchors[[6, 7, 8]])
    o2 = yolo_boxes(outputs[1], yolo_anchors[[3, 4, 5]])
    o3 = yolo_boxes(outputs[2], yolo_anchors[[0, 1, 2]])

    bbox = tf.concat([tf.reshape(o1[0], (tf.shape(o1[0])[0], -1,
                                         tf.shape(o1[0])[-1])),
                      tf.reshape(o2[0], (tf.shape(o2[0])[0], -1,
                                         tf.shape(o2[0])[-1])),
                      tf.reshape(o3[0], (tf.shape(o3[0])[0], -1,
                                         tf.shape(o3[0])[-1]))],
                     axis=1)

    confidence = tf.concat([tf.reshape(o1[1], (tf.shape(o1[1])[0], -1,
                                               tf.shape(o1[1])[-1])),
                            tf.reshape(o2[1], (tf.shape(o2[1])[0], -1,
                                               tf.shape(o2[1])[-1])),
                            tf.reshape(o3[1], (tf.shape(o3[1])[0], -1,
                                               tf.shape(o3[1])[-1]))],
                           axis=1)

    class_probs = tf.concat([tf.reshape(o1[2], (tf.shape(o1[2])[0], -1,
                                                tf.shape(o1[2])[-1])),
                             tf.reshape(o2[2], (tf.shape(o2[2])[0], -1,
                                                tf.shape(o2[2])[-1])),
                             tf.reshape(o3[2], (tf.shape(o3[2])[0], -1,
                                                tf.shape(o3[2])[-1]))],
                            axis=1)

    scores = confidence * class_probs
    boxes, scores, classes, valid_detections = tf.combined_non_max_suppression(
        boxes=tf.reshape(bbox, (bbox.shape[0], -1, 1, 4)),
        scores=tf.reshape(scores, (scores.shape[0], -1, scores.shape[-1])),
        max_output_size_per_class=20,
        max_total_size=20,
        iou_threshold=0.5,
        score_threshold=0.5
    )

    return boxes, scores, classes, valid_detections
