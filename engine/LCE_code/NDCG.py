import numpy as np
from numpy import array, r_
import math
import pandas as pd
import scipy as sp
import scipy.sparse

def NDCG(p,y):
    n, m = np.shape(p)
    num_rel = np.sum(y,2)
    void, idx = np.argsort(-p, kind='quicksort')
    result = 0
    denominator = np.array([1, 1/math.log(m[2:m], 2)])

    for i in range(1, n):
        DCG = np.sum(y(i, idx[i,:]) * denominator)
        IDCG = np.sum(np.array(1, 1/math.log(num_rel[2:num_rel(i)], 2)))
        result = result + ((DCG / IDCG) / n)

    return result