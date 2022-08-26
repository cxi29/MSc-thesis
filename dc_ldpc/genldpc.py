import numpy as np
from pyldpc import make_ldpc, parity_check_matrix

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