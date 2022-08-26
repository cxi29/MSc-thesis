""" 
An implementation of Drop-Connect Layer in tensorflow 2.x. Implementation of layers of Dense, Conv2D, and Wrapper(for all TensorFlow Layers) has been done.
Source: https://github.com/AryaAftab/dropconnect-tensorflow.git
"""
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.layers import Dense, Conv2D, Wrapper
from tensorflow.keras.layers import ReLU, BatchNormalization, Flatten, MaxPool2D, Input




#classes

class DropConnectDense(Dense):
    def __init__(self, *args, **kwargs):
        self.prob = kwargs.pop('prob', 0.5)
        if not 0. <= self.prob < 1.:
            raise NameError('prob must be at range [0, 1)]')


        super(DropConnectDense, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self.in_feature = input_shape[-2]
        # print("in_feature: %s" % self.in_feature)
        super(DropConnectDense, self).build(input_shape)

    def call(self, inputs, train=False):
        if train:
            self.kernel = K.in_train_phase(K.dropout(self.kernel, self.prob) * (1-self.prob), self.kernel)
            self.bias = K.in_train_phase(K.dropout(self.bias, self.prob) * (1-self.prob), self.bias)
            output = tf.matmul(inputs, self.kernel)
        else:
            output = tf.matmul(inputs, self.kernel)
          
        if self.use_bias:
            output += self.bias
        return self.activation(output)
        
        
        
        
        
        
class DropConnectConv2D(Conv2D):
    def __init__(self, *args, **kwargs):
        self.prob = kwargs.pop('prob', 0.5)
        if not 0. <= self.prob < 1.:
            raise NameError('prob must be at range [0, 1)]')

        super(DropConnectConv2D, self).__init__(*args, **kwargs)

        if type(self.padding) is str:
            self.padding = self.padding.upper()

        
    def build(self, input_shape):
        self.in_channel = input_shape[-1]
        
        super(DropConnectConv2D, self).build(input_shape)

    def call(self, inputs, train=False):
        if train:
            mask = tf.cast(tf.random.uniform((self.kernel_size[0], self.kernel_size[1], self.self.in_channel, self.filters)) <= self.prob, tf.float32)
            kernel = tf.multiply(self.kernel, mask)
            output = tf.nn.conv2d(inputs,
                                  kernel,
                                  strides=self.strides,
                                  padding=self.padding,
                                  dilations=self.dilation_rate)
        else:
            output = tf.nn.conv2d(inputs,
                                  self.kernel * (1 - self.prob),
                                  strides=self.strides,
                                  padding=self.padding,
                                  dilations=self.dilation_rate)
          
        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)
        return self.activation(output)





class DropConnect(Wrapper):
    def __init__(self, layer, prob=0.0, **kwargs):

        if not 0. <= prob < 1.:
            raise NameError('prob must be at range [0, 1)]')

        self.prob = prob
        self.layer = layer
  
        super(DropConnect, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True

        self.n_trainable = len(self.layer.trainable_weights)

        super(DropConnect, self).build()

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def call(self, x):
        for counter in range(self.n_trainable):
            self.layer.trainable_weights[counter] = K.in_train_phase(K.dropout(self.layer.trainable_weights[counter], self.prob) * (1 - self.prob),
                                                                     self.layer.trainable_weights[counter])
        return self.layer.call(x)