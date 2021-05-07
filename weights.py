import numpy as np
from yolov3_tf2.models import YoloV3
from yolov3_tf2.utils import load_darknet_weights
import tensorflow as tf
import valohai

params = {
    "weights_num_classes": 80,
}

inputs = {
    "weights": "https://pjreddie.com/media/files/yolov3.weights",
}

valohai.prepare(step="weights", default_parameters=params, default_inputs=inputs)


def main():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    yolo = YoloV3(classes=valohai.parameters('weights_num_classes').value)
    yolo.summary()
    print('model created')

    load_darknet_weights(yolo, valohai.inputs('weights').path(), False)
    print('weights loaded')

    img = np.random.random((1, 320, 320, 3)).astype(np.float32)
    output = yolo(img)
    print('sanity check passed')

    path = valohai.outputs('model').path('model.tf')
    yolo.save_weights(path)
    print(f'weights saved {path}')


if __name__ == "__main__":
    main()
