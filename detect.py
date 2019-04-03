from absl import app, flags, logging
from absl.flags import FLAGS
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import YoloV3, YoloV3Tiny, yolo_output

flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './data/yolov3.h5', 'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_string('image', './data/girl.png', 'path to input image')
flags.DEFINE_string('output', './data/output.jpg', 'path to output image')


def main(_argv):
    if FLAGS.tiny:
        yolo = YoloV3Tiny()
    else:
        yolo = YoloV3()

    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    img = tf.image.decode_image(open(FLAGS.image, 'rb').read(), channels=3)
    img = tf.image.resize(img, (416, 416))
    img = img / 255.0
    img = tf.expand_dims(img, 0)

    boxes, scores, classes, nums = yolo_output(yolo((img)))

    for i in range(nums[0]):
        logging.info('{}, {}, {}'.format(class_names[classes[0][i]],
                                         scores[0][i].numpy(),
                                         boxes[0][i].numpy()))


if __name__ == '__main__':
    app.run(main)
