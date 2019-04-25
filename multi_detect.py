# using function in function is strange!!!!!


import matplotlib.pyplot as plt
import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)


from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs



images_url=["./data/meme.jpg","street.jpg","./data/girl.png"]

  
  
flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_string('output_file', '', 'path to output image')


def main(_argv):
    del _argv
    if FLAGS.tiny:
        yolo = YoloV3Tiny()
    else:
        yolo = YoloV3()
#    yolo.summary()
    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')
    
    url=images_url.copy()
    img=url[0]
    img = tf.image.decode_image(open(img, 'rb').read(), channels=3)
    img = tf.expand_dims(img, 0)  
    im= transform_images(img, 416)
    
    for pic in url[1:]:
        
        img1 = tf.image.decode_image(open(pic, 'rb').read(), channels=3)
        img1 = tf.expand_dims(img1, 0)  
        img1= transform_images(img1, 416)
        im=tf.concat((im,img1),axis=0)

    print(tf.shape(im))


    t1 = time.time()
    Tboxes, Tscores, Tclasses, Tnums = yolo(im)
    t2 = time.time()
    logging.info('time: {}'.format(t2 - t1))
#    boxes, scores, classes, nums=a
#    print(tf.shape(Tscores[0:1,:]))
#    print(tf.shape(Tclasses))
#    print(tf.shape(Tboxes))
#    print(tf.shape(Tnums[0:1]))



    for pic in range(tf.shape(Tnums)):
        scores=Tscores[0+pic:1+pic,:]
        classes=Tclasses[0+pic:1+pic,:]
        boxes=Tboxes[0+pic:1+pic,:,:]
        nums=Tnums[0+pic:1+pic]

        
        logging.info('detections:')
        print(nums[0])
        for i in range(nums[0]):
            logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                           scores[0][i].numpy(),
                                           boxes[0][i].numpy()))

        
        img = cv2.imread(images_url[pic])
        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        cv2.imwrite(str(pic)+'.jpg', img)
        logging.info('output saved to: {}'.format('output'+str(pic)))

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

