import numpy as np
from numpy import array, r_
import math
import pandas as pd
import scipy as sp
import scipy.sparse

def sparse(i, j, v, m, n):
    """
    Create and compressing a matrix that have many zeros
    Parameters:
        i: 1-D array representing the index 1 values
            Size n1
        j: 1-D array representing the index 2 values
            Size n1
        v: 1-D array representing the values
            Size n1
        m: integer representing x size of the matrix >= n1
        n: integer representing y size of the matrix >= n1
    Returns:
        s: 2-D array
            Matrix full of zeros excepting values v at indexes i, j
    """
    return scipy.sparse.csr_matrix((v, (i, j)), shape=(m, n))

def construct_A(x, k, binary):
    n = x.shape[0]

    # x = L2_norm_row(x)
    s = x * x

    vals, inds =  np.argsort(-s, kind='quicksort')
    # vals = vals(:, 2:(k+1))
    vals = vals[r_[:, 2:(k+1)]]
    inds = inds[r_[:, 2:(k+1)]]

    r = [np.tile(r_[1:n], (vals[:], inds[:]))]
    a = sparse(r[:, 1], r[:, 2], r[:, 3], n, n)
    a = np.maximum(a, a.conj().T)
    if binary:
        for x in a:
            if x > 0:
                x = 1
        return a

    return a
        # no_duplicates = [list(v) for v in dict(movies).items()]
        # a = [x for x in a]
        # [] = 1