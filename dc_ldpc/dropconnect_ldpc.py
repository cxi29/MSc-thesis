import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.layers import Dense, Conv2D, Wrapper
from tensorflow.keras.layers import ReLU, BatchNormalization, Flatten, MaxPool2D, Input

from dc_ldpc.genldpc import parity_check_matrix

def gen_ldpc (n, m, prob):
    """ 
    :param n: Nr. of columns in ldpc matrix (input features in neural network).
    :param m: Nr. of rows in ldpc matrix, m = n - k (output units in neural network).
    :param prob: Dropout probability p.
    """
    # seed = np.random.RandomState(826)
    dv = int((1-prob) * m)
    dc = int((1-prob) * n)
    # H = make_ldpc(n, dv, dc, seed=seed, sparse=True)
    H = parity_check_matrix(n, m, dv, dc)
    return H

class LDPC_DropConnectDense(Dense):
    def __init__(self, **kwargs):
        self.prob = kwargs.pop('prob')
        if not 0. <= self.prob < 1.:
            raise NameError('prob must be at range [0, 1)]')
        super(LDPC_DropConnectDense, self).__init__(**kwargs)

    def build(self, input_shape): 
        self.in_feature = input_shape[-1]

        super(LDPC_DropConnectDense, self).build(input_shape)

    def call(self, inputs):
        
        ldpc_mask = gen_ldpc(self.in_feature, self.units, self.prob).T
        ddmask = tf.cast(ldpc_mask, tf.float32)       
        self.kernel = K.in_train_phase(tf.multiply(self.kernel, ddmask), self.kernel)        
        # self.kernel = K.in_train_phase(tf.multiply(self.kernel,
        #     tf.cast(gen_ldpc(self.units, self.in_feature, self.prob), tf.float32)), self.kernel)
        # self.bias = K.in_train_phase(tf.multiply(self.bias, mask), self.bias)        
        output = tf.matmul(inputs, self.kernel)
          
        if self.use_bias:
            output += self.bias
        return self.activation(output)        

        # if self.train:
        #     ldpc_mask = gen_ldpc(self.units, self.in_feature, self.prob)
        #     mask = tf.cast(ldpc_mask, tf.float32)
        #     self.kernel = tf.multiply(self.kernel, mask) * (1-self.prob)
        #     self.bias = tf.multiply(self.bias, mask) * (1-self.prob)
        #     output = tf.matmul(inputs, self.kernel)
        # else:
        #     output = tf.matmul(inputs, self.kernel) * (1-self.prob)    
        # if self.use_bias:
        #     output += self.bias
        # return self.activation(output)          
        



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
        self.n_trainable = len(self.layer.trainable_weights)
        super(LDPC_DropConnect, self).build()

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def call(self, x):
        for counter in range(self.n_trainable):
            self.layer.trainable_weights[counter] = K.in_train_phase(K.dropout(self.layer.trainable_weights[counter], self.prob) * (1-self.prob),
                                                                     self.layer.trainable_weights[counter])
        return self.layer.call(x)

