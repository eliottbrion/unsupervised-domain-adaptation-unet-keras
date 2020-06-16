# Adapted from https://stackoverflow.com/questions/56841166/how-to-implement-gradient-reversal-layer-in-tf-2-0

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers

@tf.custom_gradient
def grad_reverse(x,l):
    y = tf.identity(x)
    def custom_grad(dy):
        return -dy * l, None
    return y, custom_grad

class GradReverse(Layer):
    #def __init__(self, l, **kwargs):
    def __init__(self, l):
        super().__init__()
        self.l = tf.Variable(l, trainable=False)

    def call(self, x):
        return grad_reverse(x,self.l)

    def get_config(self):
        return {'l': self.l.numpy()}
