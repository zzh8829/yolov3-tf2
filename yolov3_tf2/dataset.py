import tensorflow as tf

from .utils import float_feature, int_feature, string_feature


@tf.function
def transform_targets_for_output(y_true, grid_size, anchor_idxs):
    # y_true: (N, boxes, (x1, y1, x2, y2, class, best_anchor))
    N = tf.shape(y_true)[0]

    # y_true_out: (N, grid, grid, anchors, [x1, y1, x2, y2, obj, class])
    y_true_out = tf.zeros(
        (N, grid_size, grid_size, tf.shape(anchor_idxs)[0], 6)
    )

    anchor_idxs = tf.cast(anchor_idxs, tf.int32)

    indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
    updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
    idx = 0
    for i in tf.range(N):
        for j in tf.range(tf.shape(y_true)[1]):
            if tf.equal(y_true[i][j][2], 0):
                continue
            anchor_eq = tf.equal(
                anchor_idxs, tf.cast(y_true[i][j][5], tf.int32)
            )

            if tf.reduce_any(anchor_eq):
                box = y_true[i][j][0:4]
                box_xy = (y_true[i][j][0:2] + y_true[i][j][2:4]) / 2

                anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)
                grid_xy = tf.cast(box_xy // (1 / grid_size), tf.int32)

                # grid[y][x][anchor] = (tx, ty, bw, bh, obj, class)
                indexes = indexes.write(
                    idx, [i, grid_xy[1], grid_xy[0], anchor_idx[0][0]]
                )
                updates = updates.write(
                    idx, [box[0], box[1], box[2], box[3], 1, y_true[i][j][4]]
                )
                idx += 1

    return tf.tensor_scatter_nd_update(
        y_true_out, indexes.stack(), updates.stack()
    )


def transform_targets(y_train, anchors, anchor_masks, size):
    y_outs = []
    grid_size = size // 32

    # calculate anchor index for true boxes
    anchors = tf.cast(anchors, tf.float32)
    anchor_area = anchors[..., 0] * anchors[..., 1]
    box_wh = y_train[..., 2:4] - y_train[..., 0:2]
    box_wh = tf.tile(
        tf.expand_dims(box_wh, -2), (1, 1, tf.shape(anchors)[0], 1)
    )
    box_area = box_wh[..., 0] * box_wh[..., 1]
    intersection = tf.minimum(box_wh[..., 0], anchors[..., 0]) * tf.minimum(
        box_wh[..., 1], anchors[..., 1]
    )
    iou = intersection / (box_area + anchor_area - intersection)
    anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.float32)
    anchor_idx = tf.expand_dims(anchor_idx, axis=-1)

    y_train = tf.concat([y_train, anchor_idx], axis=-1)

    for anchor_idxs in anchor_masks:
        y_outs.append(
            transform_targets_for_output(y_train, grid_size, anchor_idxs)
        )
        grid_size *= 2

    return tuple(y_outs)


def transform_images(x_train, size):
    x_train = tf.image.resize(x_train, (size, size))
    x_train = x_train / 255
    return x_train


PARSING_CONFIG = {
    "image/filename": string_feature(),
    "image/encoded": string_feature(),
    "image/width": int_feature(),
    "image/height": int_feature(),
    "image/bbox/xmin": float_feature(fixed_len=False),
    "image/bbox/ymin": float_feature(fixed_len=False),
    "image/bbox/xmax": float_feature(fixed_len=False),
    "image/bbox/ymax": float_feature(fixed_len=False),
    "image/bbox/labels": int_feature(fixed_len=False),
}


def parse_tfrecord(tfrecord):
    x = tf.io.parse_single_example(tfrecord, PARSING_CONFIG)
    x_train = tf.image.decode_png(x["image/encoded"], channels=3)
    labels = tf.sparse.to_dense(x["image/bbox/labels"])
    y_train = tf.stack(
        [
            tf.sparse.to_dense(x["image/bbox/xmin"]),
            tf.sparse.to_dense(x["image/bbox/ymin"]),
            tf.sparse.to_dense(x["image/bbox/xmax"]),
            tf.sparse.to_dense(x["image/bbox/ymax"]),
            tf.cast(labels, dtype=tf.float32),
        ],
        axis=1,
    )
    yolo_max_boxes = 100
    paddings = [[0, yolo_max_boxes - tf.shape(y_train)[0]], [0, 0]]
    y_train = tf.pad(y_train, paddings)
    return x_train, y_train


def load_tfrecord_dataset(tfrecords_path: str):
    dataset = tf.data.TFRecordDataset(tfrecords_path)
    return dataset.map(lambda x: parse_tfrecord(x))
