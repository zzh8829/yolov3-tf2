import numpy as np
import tensorflow as tf
import cv2

# I handcrafted this list based on the graph exported from our model
# I know this is not ideal, but I couldn't find a better non-invasive solution
# to load darknet weights into tensorflow :(

YOLOV3_WEIGHTS_LIST = [
    'conv2d',
    'batch_normalization',
    'conv2d_1',
    'batch_normalization_1',
    'leaky_re_lu_1',
    'conv2d_2',
    'batch_normalization_2',
    'leaky_re_lu_2',
    'conv2d_3',
    'batch_normalization_3',
    'leaky_re_lu_3',
    'add',
    'zero_padding2d_1',
    'conv2d_4',
    'batch_normalization_4',
    'leaky_re_lu_4',
    'conv2d_5',
    'batch_normalization_5',
    'leaky_re_lu_5',
    'conv2d_6',
    'batch_normalization_6',
    'leaky_re_lu_6',
    'add_1',
    'conv2d_7',
    'batch_normalization_7',
    'leaky_re_lu_7',
    'conv2d_8',
    'batch_normalization_8',
    'leaky_re_lu_8',
    'add_2',
    'zero_padding2d_2',
    'conv2d_9',
    'batch_normalization_9',
    'leaky_re_lu_9',
    'conv2d_10',
    'batch_normalization_10',
    'leaky_re_lu_10',
    'conv2d_11',
    'batch_normalization_11',
    'leaky_re_lu_11',
    'add_3',
    'conv2d_12',
    'batch_normalization_12',
    'leaky_re_lu_12',
    'conv2d_13',
    'batch_normalization_13',
    'leaky_re_lu_13',
    'add_4',
    'conv2d_14',
    'batch_normalization_14',
    'leaky_re_lu_14',
    'conv2d_15',
    'batch_normalization_15',
    'leaky_re_lu_15',
    'add_5',
    'conv2d_16',
    'batch_normalization_16',
    'leaky_re_lu_16',
    'conv2d_17',
    'batch_normalization_17',
    'leaky_re_lu_17',
    'add_6',
    'conv2d_18',
    'batch_normalization_18',
    'leaky_re_lu_18',
    'conv2d_19',
    'batch_normalization_19',
    'leaky_re_lu_19',
    'add_7',
    'conv2d_20',
    'batch_normalization_20',
    'leaky_re_lu_20',
    'conv2d_21',
    'batch_normalization_21',
    'leaky_re_lu_21',
    'add_8',
    'conv2d_22',
    'batch_normalization_22',
    'leaky_re_lu_22',
    'conv2d_23',
    'batch_normalization_23',
    'leaky_re_lu_23',
    'add_9',
    'conv2d_24',
    'batch_normalization_24',
    'leaky_re_lu_24',
    'conv2d_25',
    'batch_normalization_25',
    'leaky_re_lu_25',
    'add_10',
    'zero_padding2d_3',
    'conv2d_26',
    'batch_normalization_26',
    'leaky_re_lu_26',
    'conv2d_27',
    'batch_normalization_27',
    'leaky_re_lu_27',
    'conv2d_28',
    'batch_normalization_28',
    'leaky_re_lu_28',
    'add_11',
    'conv2d_29',
    'batch_normalization_29',
    'leaky_re_lu_29',
    'conv2d_30',
    'batch_normalization_30',
    'leaky_re_lu_30',
    'add_12',
    'conv2d_31',
    'batch_normalization_31',
    'leaky_re_lu_31',
    'conv2d_32',
    'batch_normalization_32',
    'leaky_re_lu_32',
    'add_13',
    'conv2d_33',
    'batch_normalization_33',
    'leaky_re_lu_33',
    'conv2d_34',
    'batch_normalization_34',
    'leaky_re_lu_34',
    'add_14',
    'conv2d_35',
    'batch_normalization_35',
    'leaky_re_lu_35',
    'conv2d_36',
    'batch_normalization_36',
    'leaky_re_lu_36',
    'add_15',
    'conv2d_37',
    'batch_normalization_37',
    'leaky_re_lu_37',
    'conv2d_38',
    'batch_normalization_38',
    'leaky_re_lu_38',
    'add_16',
    'conv2d_39',
    'batch_normalization_39',
    'leaky_re_lu_39',
    'conv2d_40',
    'batch_normalization_40',
    'leaky_re_lu_40',
    'add_17',
    'conv2d_41',
    'batch_normalization_41',
    'leaky_re_lu_41',
    'conv2d_42',
    'batch_normalization_42',
    'leaky_re_lu_42',
    'add_18',
    'zero_padding2d_4',
    'conv2d_43',
    'batch_normalization_43',
    'leaky_re_lu_43',
    'conv2d_44',
    'batch_normalization_44',
    'leaky_re_lu_44',
    'conv2d_45',
    'batch_normalization_45',
    'leaky_re_lu_45',
    'add_19',
    'conv2d_46',
    'batch_normalization_46',
    'leaky_re_lu_46',
    'conv2d_47',
    'batch_normalization_47',
    'leaky_re_lu_47',
    'add_20',
    'conv2d_48',
    'batch_normalization_48',
    'leaky_re_lu_48',
    'conv2d_49',
    'batch_normalization_49',
    'leaky_re_lu_49',
    'add_21',
    'conv2d_50',
    'batch_normalization_50',
    'leaky_re_lu_50',
    'conv2d_51',
    'batch_normalization_51',
    'leaky_re_lu_51',
    'add_22',
    # darknet 53
] + [
    'conv2d_52',
    'batch_normalization_52',
    'leaky_re_lu_52',
    'conv2d_53',
    'batch_normalization_53',
    'leaky_re_lu_53',
    'conv2d_54',
    'batch_normalization_54',
    'leaky_re_lu_54',
    'conv2d_55',
    'batch_normalization_55',
    'leaky_re_lu_55',
    'conv2d_56',
    'batch_normalization_56',
    'leaky_re_lu_56',
    # yolo
    'conv2d_57',
    'batch_normalization_57',
    'leaky_re_lu_57',
    'conv2d_58',
] + [
    'conv2d_59',
    'batch_normalization_58',
    'leaky_re_lu_58',
    'up_sampling2d',
    'concatenate',
    'conv2d_60',
    'batch_normalization_59',
    'leaky_re_lu_59',
    'conv2d_61',
    'batch_normalization_60',
    'leaky_re_lu_60',
    'conv2d_62',
    'batch_normalization_61',
    'leaky_re_lu_61',
    'conv2d_63',
    'batch_normalization_62',
    'leaky_re_lu_62',
    'conv2d_64',
    'batch_normalization_63',
    'leaky_re_lu_63',
    # yolo 1
    'conv2d_65',
    'batch_normalization_64',
    'leaky_re_lu_64',
    'conv2d_66',
] + [
    'conv2d_67',
    'batch_normalization_65',
    'leaky_re_lu_65',
    'up_sampling2d_1',
    'concatenate_1',
    'conv2d_68',
    'batch_normalization_66',
    'leaky_re_lu_66',
    'conv2d_69',
    'batch_normalization_67',
    'leaky_re_lu_67',
    'conv2d_70',
    'batch_normalization_68',
    'leaky_re_lu_68',
    'conv2d_71',
    'batch_normalization_69',
    'leaky_re_lu_69',
    'conv2d_72',
    'batch_normalization_70',
    'leaky_re_lu_70',
    # yolo 2
    'conv2d_73',
    'batch_normalization_71',
    'leaky_re_lu_71',
    'conv2d_74',
]

YOLOV3_TINY_WEIGHTS_LIST = [

]


def load_yolov3_from_darknet(model, weights_file, tiny=False):
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

    if tiny:
        weights = YOLOV3_TINY_WEIGHTS_LIST
    else:
        weights = YOLOV3_WEIGHTS_LIST

    for name, i in enumerate(weights):
        layer = model.get_layer(name)
        batch_norm = None

        if i + 1 < len(weights) and weights[i+1].startswith('batch_norm'):
            batch_norm = model.get_layer(weights[i+1])

        if name.startswith('conv2d'):
            filters = layer.filters
            size = layer.kernel_size[0]
            in_dim = layer.input_shape[-1]

            if batch_norm is None:
                conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)
            else:
                # darknet [beta, gamma, mean, variance]
                bn_weights = np.fromfile(
                    wf, dtype=np.float32, count=4 * filters)
                # tf [gamma, beta, mean, variance]
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]

            # darknet shape (out_dim, in_dim, height, width)
            conv_shape = (filters, in_dim, size, size)
            conv_weights = np.fromfile(
                wf, dtype=np.float32, count=np.product(conv_shape))
            # tf shape (height, width, in_dim, out_dim)
            conv_weights = conv_weights.reshape(
                conv_shape).transpose([2, 3, 1, 0])

            if batch_norm is None:
                layer.set_weights([conv_weights, conv_bias])
            else:
                layer.set_weights([conv_weights])
                batch_norm.set_weights(bn_weights)

    assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()


def broadcast_iou(box_1, box_2):
    # box_1: (..., (x1, y1, x2, y2))
    # box_2: (N, (x1, y1, x2, y2))

    # broadcast boxes
    box_1 = tf.expand_dims(box_1, -2)
    box_2 = tf.expand_dims(box_2, 0)
    # new_shape: (..., N, (x1, y1, x2, y2))
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
    box_1 = tf.broadcast_to(box_1, new_shape)
    box_2 = tf.broadcast_to(box_2, new_shape)

    int_w = tf.maximum(tf.minimum(box_1[..., 2], box_2[..., 2]) -
                       tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
    int_h = tf.maximum(tf.minimum(box_1[..., 3], box_2[..., 3]) -
                       tf.maximum(box_1[..., 1], box_2[..., 1]), 0)
    int_area = int_w * int_h
    box_1_area = (box_1[..., 2] - box_1[..., 0]) * \
        (box_1[..., 3] - box_1[..., 1])
    box_2_area = (box_2[..., 2] - box_2[..., 0]) * \
        (box_2[..., 3] - box_2[..., 1])
    return int_area / (box_1_area + box_2_area - int_area)


def draw_outputs(img, outputs, class_names):
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])
    for i in range(nums):
        x1y1 = tuple((boxes[i][0:2].numpy() * wh).astype(np.int32))
        x2y2 = tuple((boxes[i][2:4].numpy() * wh).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
        img = cv2.putText(img, '{} {:.4f}'.format(
            class_names[classes[i]], objectness[i]),
            x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    return img
