import numpy as np
import math
import pandas as pd
import scipy as sp
import scipy.sparse

def L2_norm_row(X):
    return sp.sparse.spdiags(1. / (np.sqrt(sum(X * X, 2)) + eps), 0, len(X), len(X)) * X