import numpy as np
from numpy import array, r_
import math
import pandas as pd
import scipy as sp
import scipy.sparse
from sklearn.neighbors import NearestNeighbors

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

def construct_A(X, k, binary=False):

    nbrs = NearestNeighbors(n_neighbors=1 + k).fit(X)
    if binary:
        return nbrs.kneighbors_graph(X)
    else:
        return nbrs.kneighbors_graph(X, mode='distance')
        # no_duplicates = [list(v) for v in dict(movies).items()]
        # a = [x for x in a]
        # [] = 1