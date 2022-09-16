import numpy as np
from pyldpc import make_ldpc, parity_check_matrix

# def gen_ldpc (n, m, prob):
#     """ 
#     :param n: Nr. of columns in ldpc matrix (input features in neural network).
#     :param m: Nr. of rows in ldpc matrix, m = n - k (output units in neural network).
#     :param prob: Dropout probability p.
#     """
#     # seed = np.random.RandomState(826)
#     dv = (1-prob) * m
#     dc = (1-prob) * n
#     assert(n*dv == m*dc)
#     # H = make_ldpc(n, dv, dc, seed=seed, sparse=True)
#     H = parity_check_matrix(n, dv, dc)
#     return H

def gen_ldpc(n, m, dv):
    dc = int(n * dv / m)
    p = dv / m
    assert(n*dv == m*dc)
    H = parity_check_matrix(n, dv, dc)
    return H, p

# def gen_ldpc(n, k, dv):
#     dc = n * dv / (n-k)
#     assert(n * dv == (n-k) * dc)
#     H = np.zeros(n-k, n)
#     ones_at_rows = np.zeros(n-k, 1)
#     for i in range(1, n+1):
#         rows = np.arange(1, n-k+1)
#         rows = rows(ones_at_rows < dc)
#         # shuffle rows
#         rows = np.random.permutation(rows)


