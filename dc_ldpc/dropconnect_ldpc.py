import numpy as np
from pyldpc import make_ldpc, parity_check_matrix

import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.layers import Dense, Conv2D, Wrapper
from tensorflow.keras.layers import ReLU, BatchNormalization, Flatten, MaxPool2D, Input

# from genldpc import gen_ldpc

def gen_ldpc (n, m, prob):
    """ 
    :param n: Nr. of columns in ldpc matrix (input features in neural network).
    :param m: Nr. of rows in ldpc matrix, m = n - k (output units in neural network).
    :param prob: Dropout probability p.
    """
    # seed = np.random.RandomState(826)
    dv = (1-prob) * m
    dc = (1-prob) * n
    # H = make_ldpc(n, dv, dc, seed=seed, sparse=True)
    H = parity_check_matrix(n, dv, dc)
    return H

class LDPC_DropConnectDense(Dense):
    def __init__(self, *args, **kwargs):
        self.prob = kwargs.pop('prob', 0.5)
        if not 0. <= self.prob < 1.:
            raise NameError('prob must be at range [0, 1)]')
        super(LDPC_DropConnectDense, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self.in_feature = input_shape[-2]
        super(LDPC_DropConnectDense, self).build(input_shape)

    def call(self, inputs):
        if self.train:
            # mask = tf.cast(tf.random.uniform((self.in_feature, self.units)) <= self.prob, tf.float32)
            ldpc_mask = gen_ldpc(self.units, self.in_feature, self.prob)
            # self.mask = tf.cast(H_mask, tf.float32)
            mask = tf.cast(ldpc_mask, tf.float32)
            self.kernel = tf.multiply(self.kernel, mask)
            self.bias = tf.multiply(self.bias, mask)
            output = tf.matmul(inputs, self.kernel)
        else:
            output = tf.matmul(inputs, self.kernel)
          
        if self.use_bias:
            output += self.bias
        return self.activation(output)
        
        
        
        
        
        
# class DropConnectConv2D(Conv2D):
#     def __init__(self, *args, **kwargs):
#         self.prob = kwargs.pop('prob', 0.5)
#         if not 0. <= self.prob < 1.:
#             raise NameError('prob must be at range [0, 1)]')

#         super(DropConnectConv2D, self).__init__(*args, **kwargs)

#         if type(self.padding) is str:
#             self.padding = self.padding.upper()

        
#     def build(self, input_shape):
#         self.in_channel = input_shape[-1]
        
#         super(DropConnectConv2D, self).build(input_shape)

#     def call(self, inputs, train=False):
#         if train:
#             mask = tf.cast(tf.random.uniform((self.kernel_size[0], self.kernel_size[1], self.self.in_channel, self.filters)) <= self.prob, tf.float32)
#             kernel = tf.multiply(self.kernel, mask)
#             output = tf.nn.conv2d(inputs,
#                                   kernel,
#                                   strides=self.strides,
#                                   padding=self.padding,
#                                   dilations=self.dilation_rate)
#         else:
#             output = tf.nn.conv2d(inputs,
#                                   self.kernel * (1 - self.prob),
#                                   strides=self.strides,
#                                   padding=self.padding,
#                                   dilations=self.dilation_rate)
          
#         if self.use_bias:
#             output = tf.nn.bias_add(output, self.bias)
#         return self.activation(output)





class LDPC_DropConnect(Wrapper):
    def __init__(self, layer, prob=0.0, **kwargs):

        if not 0. <= prob < 1.:
            raise NameError('prob must be at range [0, 1)]')
        self.prob = prob
        self.layer = layer
        super(LDPC_DropConnect, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        # self.n_trainable = len(self.layer.trainable_weights)
        super(LDPC_DropConnect, self).build()

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    # def call(self, x):
    #     for counter in range(self.n_trainable):
    #         self.layer.trainable_weights[counter] = K.in_train_phase(K.dropout(self.layer.trainable_weights[counter], self.prob) * (1-self.prob),
    #                                                                  self.layer.trainable_weights[counter])
    #     return self.layer.call(x)

    def call(self, inputs):
        if self.train:
            ldpc_mask = gen_ldpc(self.units, self.in_feature, self.prob)
            mask = tf.cast(ldpc_mask, tf.float32)
            self.kernel = tf.multiply(self.kernel, mask)
            self.bias = tf.multiply(self.bias, mask)
            output = tf.matmul(inputs, self.kernel)
        else:
            output = tf.matmul(inputs, self.kernel)
        if self.use_bias:
            output += self.bias
        return self.layer.call(output)
