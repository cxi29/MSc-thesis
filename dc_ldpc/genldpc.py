# import numpy as np
from pyldpc import coding_matrix_systematic

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

# def gen_ldpc(n, m, dv):
#     dc = int(n * dv / m)
#     p = dv / m
#     assert(n*dv == m*dc)
#     H = parity_check_matrix(n, dv, dc)
#     return H, p

import numpy as np
from pyldpc import utils


def parity_check_matrix(n_code, n_equ, d_v, d_c, systematic=False, sparse=True, seed=None):
    """
    Inherited from pyldpc/code.py

    """
    rng = utils.check_random_state(seed)

    if d_v <= 1:
        raise ValueError("""d_v must be at least 2.""")

    if d_c <= d_v:
        raise ValueError("""d_c must be greater than d_v.""")

    # if n_code % d_c:
    #     raise ValueError("""d_c must divide n for a regular LDPC matrix H.""")

    n_equations = (n_code * d_v) // d_c
    n_equations = n_equ

    block = np.zeros((n_equations // d_v, n_code), dtype=int)
    H = np.empty((n_equations, n_code))
    H_check = np.empty((1, n_code)) # Check if  there is circle in H
    block_size = n_equations // d_v

    # Filling the first block with consecutive ones in each row of the block

    for i in range(block_size):
        for j in range(i * d_c, (i+1) * d_c):
            block[i, j] = 1
    H[:block_size] = block

    # reate remaining blocks by permutations of the first block's columns:
    for i in range(1, d_v):
        H_t = rng.permutation(block.T).T
        # c = circle_check(H_t, block_size, d_c)
        # while (c == 1):
        #     H_t = rng.permutation(block.T).T
        #     c = circle_check(H_t, block_size, d_c)
        H[i * block_size: (i + 1) * block_size] = H_t         
    H = H.astype(int)
    if systematic:
        H, _ = coding_matrix_systematic(H, sparse=sparse)    
    return H

def circle_check(H_t, block_size, d_c):
    for j in range(block_size):
        c_sum = 0
        H_check = np.sum(H_t[:, j*d_c:(j+1)*d_c], axis=1)
        for k in range(block_size):
            if (H_check[k] > 2):
                return 1
            if (H_check[k] == 2):
                c_sum = c_sum +1
                if (c_sum > 1):
                    return 1
            # if (H_check[k] > 1):
            #     return 1
    return 0

def rank_cal(H):
    return np.linalg.matrix_rank(H)

import matplotlib.pyplot as plt

def plotmat(H, idx):
    plt.matshow(H, fignum=idx, cmap=plt.cm.gray_r)

""" Generate random matrix for comparison """
def rand_bin_mat(K, N, p):   # p=K/N
    arr = np.zeros(N * K)
    n_ones = int(N*K*p)
    arr[:n_ones]  = 1
    np.random.shuffle(arr)
    rbmat = np.reshape(arr,(K,N))
    return rbmat

if __name__ == '__main__':
    n = 128
    m = 64    
    # plt.figure()
    p = [1/8, 1/4, 1/2]
    for i in range(3):
        pi = p[i]
        dv = int(pi*m)
        dc = int(pi*n)
        H = parity_check_matrix(n,m,dv,dc,systematic=True)
        M = rand_bin_mat(m,n,pi)
        plt.subplot(2,3,i+1)
        plotmat(H,False)
        plt.subplot(2,3,i+4)
        plotmat(M,False)
    plt.show()

