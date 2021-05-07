import os
import json
import hashlib
import cv2

import tensorflow as tf


def parse_bounding_box(data, image_width, image_height):
    return {
        "xmin": data['x'] / image_width,
        "ymin": data['y'] / image_height,
        "xmax": (data['x'] + data['width']) / image_width,
        "ymax": (data['y'] + data['height']) / image_height,
    }


def parse_bounding_box_labels(labels, classes, image_width, image_height):
    result = []
    for obj in labels:
        if obj['annotationType'] == 'box':
            label = parse_bounding_box(obj['annotation']['coord'], image_width, image_height)
            label['classname'] = obj['className']
            label['classid'] = classes.index(obj['className'])
            label['difficulty'] = 0
            label['truncated'] = 0
            label['view'] = ""
            result.append(label)
    return result


def get_tf_example(
        filename,
        width,
        height,
        img_raw,
        xmins,
        ymins,
        xmaxs,
        ymaxs,
        class_ids,
        class_names,
        difficulties,
        truncations,
        views):

    return tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.encode('utf8')])),
        'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.encode('utf8')])),
        'image/key/sha256': tf.train.Feature(bytes_list=tf.train.BytesList(value=[hashlib.sha256(img_raw).hexdigest().encode('utf8')])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=['jpeg'.encode('utf8')])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=[c.encode('utf8') for c in class_names])),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=class_ids)),
        'image/object/difficult': tf.train.Feature(int64_list=tf.train.Int64List(value=difficulties)),
        'image/object/truncated': tf.train.Feature(int64_list=tf.train.Int64List(value=truncations)),
        'image/object/view': tf.train.Feature(bytes_list=tf.train.BytesList(value=[v.encode('utf8') for v in views])),
        }))


def parse_labels_to_tensorflow(data, classes, images_path, labels_path):
    key = data['data_key']
    test_data = any(['name' in tag and tag['name'] == 'test' for tag in data['tags']])
    image_path = os.path.join(images_path, key)
    assert os.path.exists(image_path)

    with open(os.path.join(labels_path, data['label_path'][0])) as json_file:
        im = cv2.imread(image_path)
        width = im.shape[1]
        height = im.shape[0]
        labels_data = json.load(json_file)
        objects = labels_data['objects']
        boxes = parse_bounding_box_labels(objects, classes, width, height)
        return get_tf_example(
            filename=key,
            width=width,
            height=height,
            img_raw=open(image_path, 'rb').read(),
            xmins=[box['xmin'] for box in boxes],
            xmaxs=[box['xmax'] for box in boxes],
            ymins=[box['ymin'] for box in boxes],
            ymaxs=[box['ymax'] for box in boxes],
            class_ids=[box['classid'] for box in boxes],
            class_names=[box['classname'] for box in boxes],
            difficulties=[box['difficulty'] for box in boxes],
            truncations=[box['truncated'] for box in boxes],
            views=[box['view'] for box in boxes],
        ), test_data
