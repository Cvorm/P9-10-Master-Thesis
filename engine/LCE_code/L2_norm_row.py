import numpy as np
import math
import pandas as pd
import scipy as sp
import scipy.sparse

def L2_norm_row(x):
    xnorm = sp.sparse.diags(np.array([1 / (np.sqrt(np.sum(x * x, 2))), 0, x.shape[0], x.shape[0]])) @ x
    return xnorm