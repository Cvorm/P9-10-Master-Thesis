import numpy as np
from numpy import array, r_
import math
import pandas as pd
import scipy as sp
import scipy.sparse
import scipy.io
from construct_A import *
from LCE_Beta0 import *
import sklearn as sk
# import tfidf
import h5py
from NDCG import *
eps = 7. / 3 - 4. / 3 - 1

def L2_norm_row(X):
    "Normalize each row of the matrix."
    return sp.sparse.spdiags(1. / (np.sqrt(np.sum(X * X, axis=1)) + eps), 0, len(X), len(X)).dot(X)

def tfidf(x_train, x_test):
    "Calculate the term frequency–inverse document frequency fro the train and test sets"
    idf = np.log(np.divide(x_train.shape[0], sum(x_train > 0) + eps))
    idf = idf.reshape(idf.shape[0], 1)
    IDF = sp.sparse.spdiags(idf.conj().T, 0, idf.shape[0], idf.shape[0]).toarray()

    x_train = x_train.dot(IDF)
    x_train = L2_norm_row(x_train)

    x_test = x_test.dot(IDF)
    x_test = L2_norm_row(x_test)

    return x_train, x_test


"""
Load in the data:
nips12: NIPS Conference Papers Vols0-12 
tmat: topics of the papers (i think)
"""

nips12 = scipy.io.loadmat('../LCE_code/LCE_data/nips12raw_str602.mat', simplify_cells=True)

tmat = scipy.io.loadmat('../LCE_code/LCE_data/T.mat')



"Transforming the data into a suitable matrix form"

nd = nips12['counts'].shape[1]
nw = nips12['counts'].shape[0]
na = nips12['apapers'].shape[0]


papers = []
users = []
for x in nips12['apapers']:
    if isinstance(x, int):
        papers.append(x)
    else:
        for k in x:
            papers.append(k)


indexxx = sorted(set(papers))
user_item = pd.DataFrame(index=[x for x in range(nd+1)], columns=[x for x in range(na)]).fillna(0.0)

for user in user_item.columns:

    papers = nips12['apapers'][user]
    if isinstance(papers, int):
        user_item.at[papers, user] = 1.0
    else:
        for x in papers:
            user_item.at[x, user] = 1.0


mu = user_item.drop([0], axis=0)
user_item.to_csv("DEMOOOOOO.csv", sep='\t')
xu = scipy.sparse.csr_matrix(mu.values).toarray()
T = tmat['T'].astype(np.float32)

train_time = np.where(np.sum(T[:, 0:12], axis=1))
train_time = np.asarray(train_time[0])
train_time = train_time.reshape(train_time.shape[0], 1)


test_time = np.where(np.sum(T[:, 12], axis=1))
test_time = np.asarray(test_time[0])
test_time = test_time.reshape(test_time.shape[0], 1)

vocab = [x for x in range(nw)] #1d vector
vocab = np.array(vocab)


xu_train = xu[train_time, :]
counts = nips12['counts'].toarray().astype(np.float32)


xs_train = counts[:, train_time].conj().T

xs_train = xs_train[0,:,:]
print(xs_train)

xs_train = xs_train[:, vocab]

xu_test = xu[test_time, :]
xs_test = counts[:, test_time].conj().T
xs_test = xs_test[0,:,:]
xs_test = xs_test[:, vocab]


"Preprocessing the data"
train_authors = (sum(xu_train) > 0)
xu_train = xu_train[:, train_authors]
xu_test = xu_test[:, train_authors]

xs_train, xs_test = tfidf(xs_train, xs_test)

"Running the LCE"
k = 500
alpha = 0.5
lambdaa = 0.5
epsilon = 0.001
maxiter = 10
verbose = True
beta = 0.05

"Constructing the adjacency matrix"

a = construct_A(xs_train, 1, True)
w, hu, hs, objhistory = LCE(xs_train,  L2_norm_row(xu_train), a, k, alpha, beta, lambdaa, epsilon, maxiter, verbose)


w_test = np.dot(xs_test, np.linalg.pinv(hs))
w_test[w_test < 0] = 0
lce_rank = np.dot(w_test, hu)
lce_res = sk.metrics.ndcg_score(xu_test, lce_rank)
print(lce_res)

