import tensorflow as tf


class BatchNormalization(tf.keras.layers.BatchNormalization):
    """
    Make trainable=False freeze BN for real (the og version is sad)
    """

    def call(self, x, training=False):
        if training is None:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)


def freeze_bn(model, frozen):
    if isinstance(model, tf.keras.layers.BatchNormalization):
        model.trainable = not frozen
    elif isinstance(model, tf.keras.Model):
        for l in model.layers:
            freeze_bn(l, frozen)
