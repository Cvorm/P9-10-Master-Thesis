import numpy as np
from numpy import array, r_
import math
import pandas as pd
import scipy as sp
import scipy.sparse
import scipy.io
import h5py

# nips12 = h5py.File('C:\\Users\\caspe\\PycharmProjects\\P9-10-Master-Thesis\\engine\\LCE_code\\LCE_data\\nips12raw_str602.mat', 'r')
# tmat = h5py.File('T.mat', 'r')

# nips12 = scipy.io.loadmat('../LCE_code/LCE_data/nips12raw_str602.mat', variable_names=['counts, apapers'])
nips12 = scipy.io.loadmat('../LCE_code/LCE_data/nips12raw_str602.mat')

tmat = scipy.io.loadmat('../LCE_code/LCE_data/T.mat')

print(nips12['apapers'][1])

nd = nips12['counts'].shape[1]
nw = nips12['counts'].shape[0]
na = nips12['apapers'].shape[0]

xu = scipy.sparse.csr_matrix((nd, nw), dtype=float).toarray()
# for i in xu:
#     print(i)
# print(xu[1])
#
# for i in range(na):
#     xu[nips12['apapers'][i]][1] = 1

# print(xu)
#     xu[nips12{'apapers'}, i] = 1


# nips12.keys()
# data1 = nips12.get()
# nd = coun