import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.layers import Dense, Conv2D, Wrapper
from tensorflow.keras.layers import ReLU, BatchNormalization, Flatten, MaxPool2D, Input


from tensorflow_probability.python import random as tfp_random
from tensorflow_probability.python.distributions import independent as independent_lib
from tensorflow_probability.python.distributions import kullback_leibler as kl_lib
from tensorflow_probability.python.distributions import normal as normal_lib
from tensorflow_probability.python.layers import util as tfp_layers_util
from tensorflow_probability.python.util import SeedStream

from dc_ldpc.genldpc import parity_check_matrix

def gen_ldpc (n, m, prob):
    """ 
    :param n: Nr. of columns in ldpc matrix (input features in neural network).
    :param m: Nr. of rows in ldpc matrix, m = n - k (output units in neural network).
    :param prob: NOT dropping probability p.
    """
    # seed = np.random.RandomState(826)
    dv = int(prob* m)
    dc = int(prob * n)
    # H = make_ldpc(n, dv, dc, seed=seed, sparse=True)
    H = parity_check_matrix(n, m, dv, dc)
    return H

class LDPC_DropConnectDense(Dense):
    def __init__(self, prob, **kwargs):
        super(LDPC_DropConnectDense, self).__init__(**kwargs)
        # self.prob = kwargs.pop('prob')
        self.prob = prob
        if not 0. <= self.prob < 1.:
            raise NameError('prob must be at range [0, 1)]')

    def build(self, input_shape): 
        self.in_feature = input_shape[-1]
        # self.w = self.add_weight(
        #     shape=(self.in_feature, self.units),
        #     initializer="random_normal",
        #     trainable=True,
        #     )
        # self.b = self.add_weight(
        #     shape=(self.units, ), 
        #     initializer="random_normal", 
        #     trainable=False
        #     )
        super(LDPC_DropConnectDense, self).build(input_shape)

    def call(self, inputs, training):
        
        if training is None:
            training = K.learning_phase()    

        # if training is True:
        #     self.ddmask = gen_ldpc(self.in_feature, self.units, self.prob).T
        #     self.ddmask = tf.cast(self.ddmask, tf.float32)      
        #     self.w_masked = tf.multiply(self.kernel, self.ddmask)
        #     output = tf.matmul(inputs, self.w_masked)
        self.kernel = K.in_train_phase(tf.multiply(self.kernel, 
                                        tf.cast(gen_ldpc(self.in_feature, self.units, self.prob).T, tf.float32)),
                                        self.kernel)   
        # else:
        #     output = tf.matmul(inputs, self.kernel)
        output = tf.matmul(inputs, self.kernel)
          
        if self.use_bias:
            output += self.bias
        return self.activation(output)        
       
        



class LDPC_DropConnect_Flipout(Dense):
    def __init__(self, prob, units, seed=None,
                kernel_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(),
                kernel_posterior_tensor_fn=lambda d: d.sample(),
                kernel_prior_fn=tfp_layers_util.default_multivariate_normal_fn,
                kernel_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
                **kwargs):
        super(LDPC_DropConnect_Flipout, self).__init__(units, **kwargs)
        # self.prob = kwargs.pop('prob')
        self.prob = prob
        self.units = units
        self.kernel_posterior_fn = kernel_posterior_fn
        self.kernel_posterior_tensor_fn = kernel_posterior_tensor_fn
        self.kernel_prior_fn = kernel_prior_fn
        self.kernel_divergence_fn = kernel_divergence_fn
        self.seed = seed

        if not 0. <= self.prob < 1.:
            raise NameError('prob must be at range [0, 1)]')

    def build(self, input_shape): 
        
        self.in_feature = input_shape[-1]
        self.kernel_posterior = self.kernel_posterior_fn(tf.float32, #dtype
                                    [self.in_feature, self.units], 
                                    'kernel_posterior',
                                    self.trainable, self.add_variable)
        if self.kernel_prior_fn is None:
            self.kernel_prior = None
        else:
            self.kernel_prior = self.kernel_prior_fn(tf.float32, 
                                    [self.in_feature, self.units], 
                                    'kernel_prior',
                                    self.trainable, self.add_variable)

        # if self.bias_posterior_fn is None:
        #     self.bias_posterior = None
        # else:
        #     self.bias_posterior = self.bias_posterior_fn(tf.float32, 
        #                                                 [self.units], 
        #                                                 'bias_posterior',
        #                                                 self.trainable, self.add_variable)

        # if self.bias_prior_fn is None:
        #     self.bias_prior = None
        # else:
        #     self.bias_prior = self.bias_prior_fn(tf.float32, 
        #                                         [self.units], 'bias_prior',
        #                                         self.trainable, self.add_variable)

        self.built = True

        super(LDPC_DropConnect_Flipout, self).build(input_shape)

    def call(self, inputs, training=False):
        
        if training is None:
            training = K.learning_phase()    

        inputs = tf.convert_to_tensor(value=inputs, dtype=self.dtype)
        if training:
            self.ddmask = gen_ldpc(self.in_feature, self.units, self.prob).T
            self.ddmask = tf.cast(self.ddmask, tf.float32)
            # self.w_masked = tf.multiply(self.kernel, self.ddmask)
          
        # if self.use_bias:
        #     output += self.bias

        outputs = self._apply_variational_kernel(inputs, training)
        if self.activation is not None:
            outputs = self.activation(outputs)
        
        self._apply_divergence(
            self.kernel_divergence_fn,
            self.kernel_posterior,
            self.kernel_prior,
            self.kernel_posterior_tensor,
            name='divergence_kernel')
        
        # self._apply_divergence(
        #     self.bias_divergence_fn,
        #     self.bias_posterior,
        #     self.bias_prior,
        #     self.bias_posterior_tensor,
        #     name='divergence_bias')
        return outputs


    def _apply_variational_kernel(self, inputs, train=False):
        if (not isinstance(self.kernel_posterior, independent_lib.Independent) or
            not isinstance(self.kernel_posterior.distribution, normal_lib.Normal)):
            raise TypeError(
            '`DenseFlipout` requires '
            '`kernel_posterior_fn` produce an instance of '
            '`tfd.Independent(tfd.Normal)` '
            '(saw: \"{}\").'.format(self.kernel_posterior.name)
            )
            
        self.kernel_posterior_affine = normal_lib.Normal(
            loc=tf.zeros_like(self.kernel_posterior.distribution.loc),
            scale=self.kernel_posterior.distribution.scale)
        self.kernel_posterior_affine_tensor = (
            self.kernel_posterior_tensor_fn(self.kernel_posterior_affine))
        self.kernel_posterior_tensor = None

        if train:
            self.kernel_posterior_affine_tensor = tf.multiply(self.ddmask, self.kernel_posterior_affine_tensor)
        
        input_shape = tf.shape(inputs)
        batch_shape = input_shape[:-1]

        seed_stream = SeedStream(self.seed, salt='LDPC_DropConnect_Flipout')

        sign_input = tfp_random.rademacher(
            input_shape,
            dtype=inputs.dtype,
            seed=seed_stream())
        sign_output = tfp_random.rademacher(
            tf.concat([batch_shape,
                    tf.expand_dims(self.units, 0)], 0),
            dtype=inputs.dtype,
            seed=seed_stream())
        perturbed_inputs = tf.matmul(
            inputs * sign_input, self.kernel_posterior_affine_tensor) * sign_output

        # if train:
        #     self.kernel_posterior.distribution.loc = tf.multiply(
        #                                             self.kernel_posterior.distribution.loc,
        #                                             self.ddmask)
        # outputs = tf.matmul(inputs, self.kernel_posterior.distribution.loc)

        outputs = tf.matmul(inputs, self.kernel_posterior_affine_tensor)
        
        outputs += perturbed_inputs
        return outputs

    def _apply_divergence(self, divergence_fn, posterior, prior,
                        posterior_tensor, name):
        if (divergence_fn is None or posterior is None or prior is None):
            divergence = None
            return
        divergence = tf.identity(
            divergence_fn(posterior, prior, posterior_tensor),
                name=name)
        self.add_loss(divergence)


