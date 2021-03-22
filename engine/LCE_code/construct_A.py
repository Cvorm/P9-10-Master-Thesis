import numpy as np
from numpy import array, r_
import math
import pandas as pd
import scipy as sp
import scipy.sparse
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph

eps = 7. / 3 - 4. / 3 - 1
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

def construct_A(X, k, binary):

    nbrs = NearestNeighbors(n_neighbors=1 + k).fit(X)
    if binary:
        return nbrs.kneighbors_graph(X)
    else:
        return nbrs.kneighbors_graph(X, mode='distance')
#         # no_duplicates = [list(v) for v in dict(movies).items()]
#         # a = [x for x in a]
#         # [] = 1

def L2_norm_row(X):
    return sp.sparse.spdiags(1. / (np.sqrt(np.sum(X * X, axis=1)) + eps), 0, len(X), len(X)).dot(X)


# def construct_A(X, k, binary):
#     n = X.shape[0]
#     X = L2_norm_row(X)
#     S = X.dot(X.conj().T)
#
#     vals = np.sort(S, axis=1)
#     inds = np.argsort(S, kind='quicksort', axis=1)
#
#     vals = vals[:, 1: (k + 1)]
#     inds = vals[:, 1: (k + 1)]
#
#     vec = [x for x in range(n)]
#     myes = np.tile(vec, (1, k))
#     myes = myes.conj().T
#     # R = [np.tile(vec, (1, k)).conj.T, inds[:], vals[:]]
#     R = array([myes, inds[:], vals[:]])
#     A = sparse(R[:, 1], R[:, 2], R[:, 3], n, n)
#     A = np.maximum(A, A.conj().T)
#
#     if binary:
#         A[A > 0] = 1
#     return A
#
#     # [vals, inds] = sort(S, 2, 'descend');
#     # vals = vals(:, 2: (k + 1));
#     # inds = inds(:, 2: (k + 1));
#     #
#     # R = [repmat(1:n, 1, k)', inds(:), vals(:)];
#     # A = sparse(R(:, 1), R(:, 2), R(:, 3), n, n);
#     # A = max(A, A
#     # ');
#     #
#     # if binary,
#     # A(A > 0) = 1;
#     # end
#     #
#     # end
#
#     # nbrs = NearestNeighbors(n_neighbors=1 + k).fit(S)
#     # if binary:
#     #     return nbrs.kneighbors_graph(S)
#     # else:
#     #     return nbrs.kneighbors_graph(S, mode='distance')