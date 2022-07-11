import numpy as np
from pyldpc import make_ldpc

def gen_ldpc (n, m, prob):
    """ 
    :param n: Nr. of columns in ldpc matrix (output units in neural network).
    :param m: Nr. of rows in ldpc matrix, m = n - k (input features in neural network).
    :param prob: Dropout probability p.
    """
    seed = np.random.RandomState(711)
    dv = prob * m
    dc = dv * m / n
    H = make_ldpc(n, dv, dc, seed=seed, sparse=True)