import tensorflow as tf


class InterLayerRecurrentModel(object):
    def __init__(self,
        batch_size=16,
        activation_function=tf.sigmoid
    ):
        self.batch_size = batch_size
        self.activation = activation_function
