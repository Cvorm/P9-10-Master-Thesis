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
    return sp.sparse.spdiags(1. / (np.sqrt(np.sum(X * X, axis=1)) + eps), 0, len(X), len(X)).dot(X)
    # return sp.sparse.spdiags(1. / (np.sqrt(np.sum(X[:,0]**2) + np.sum(X[:,1]**2)) + eps), 0, X.shape[0], X.shape[0]) * X
    # return np.sqrt(np.sum(X[:,0]**2) + np.sum(X[:,1]**2))
    # return sparse.spdiags(1. / (np.sqrt(sum(X * X, 2)) + eps), 0, len(X), len(X)) * X

def tfidf(x_train, x_test):
    # idf = np.divide(x_train.shape[0], (sum(x_train > 0) + eps))
    idf = np.log(np.divide(x_train.shape[0], sum(x_train > 0) + eps))
    idf = idf.reshape(idf.shape[0], 1)
    # myes = idf.conj().T
    # print(idf.shape[0])
    # IDF = spdiags(idf', 0, size(idf,2), size(idf,2));
    IDF = sp.sparse.spdiags(idf.conj().T, 0, idf.shape[0], idf.shape[0]).toarray()

    x_train = x_train.dot(IDF)
    # print(len(x_train))
    # x_train = np.linalg.norm(x_train, axis=(0, 1))
    x_train = L2_norm_row(x_train)

    x_test = x_test.dot(IDF)
    # print(len(x_test))
    x_test = L2_norm_row(x_test)

    return x_train, x_test


# nips12 = h5py.File('C:\\Users\\caspe\\PycharmProjects\\P9-10-Master-Thesis\\engine\\LCE_code\\LCE_data\\nips12raw_str602.mat', 'r')
# tmat = h5py.File('T.mat', 'r')

# nips12 = scipy.io.loadmat('../LCE_code/LCE_data/nips12raw_str602.mat', variable_names=['counts, apapers'])
nips12 = scipy.io.loadmat('../LCE_code/LCE_data/nips12raw_str602.mat', simplify_cells=True)

tmat = scipy.io.loadmat('../LCE_code/LCE_data/T.mat')

# print(type(nips12))
# print(nips12.get())
# for x in nips12['apapers']:
#     print(x)
# for x in nips12['counts']:
#     print(x)
# # print(nips12['apapers'])

nd = nips12['counts'].shape[1]
nw = nips12['counts'].shape[0]
na = nips12['apapers'].shape[0]

# print(len(nips12['apapers']))
# user_item = pd.DataFrame(index=[x+1 for x in range(nd)], columns=[x for x in range(na)]).fillna(0)
# print(user_item.shape)
papers = []
users = []
for x in nips12['apapers']:
    if isinstance(x, int):
        papers.append(x)
    else:
        for k in x:
            papers.append(k)

# for j in nips12['counts']:
# print(nd, nw)
# print(users)
# print(sorted(papers))
# set(papers)
indexxx = sorted(set(papers))
user_item = pd.DataFrame(index=[x for x in range(nd+1)], columns=[x for x in range(na)]).fillna(0.0)
# print(user_item.head())
for user in user_item.columns:
    # print(user)
    papers = nips12['apapers'][user]
    if isinstance(papers, int):
        user_item.at[papers, user] = 1.0
    else:
        for x in papers:
            user_item.at[x, user] = 1.0
#         paper = nips12['apapers'][user]
#         # print(paper, user)
#         user_item[paper][user] = 1
#     else:
# #         # print(user)
#         for x in nips12['apapers'][user]:
#             # print(x, user)
#             user_item[x][user] = 1

mu = user_item.drop([0], axis=0)
user_item.to_csv("DEMOOOOOO.csv", sep='\t')
xu = scipy.sparse.csr_matrix(mu.values).toarray()
T = tmat['T'].astype(np.float32)
# print(T)
# print(xdxd)
# myes = T[:, 0:12]
# print(myes)
train_time = np.where(np.sum(T[:, 0:12], axis=1))
train_time = np.asarray(train_time[0])
train_time = train_time.reshape(train_time.shape[0], 1)
# print(len(train_time[0]))

test_time = np.where(np.sum(T[:, 12], axis=1))
test_time  = np.asarray(test_time[0])
test_time  = test_time.reshape(test_time.shape[0], 1)

vocab = [x for x in range(nw)] #1d vector
vocab = np.array(vocab)
# print(vocab)
# print(test_time)

xu_train = xu[train_time, :]
counts = nips12['counts'].toarray().astype(np.float32)


xs_train = counts[:, train_time].conj().T
# xs_train = xs_train[:,0,:]
xs_train = xs_train[0,:,:]
print(xs_train)
# print(len(xs_train[0]), len(xs_train[1]), len(xs_train[2]))
# print(len(vocab))
# print(len(xs_train))
xs_train = xs_train[:, vocab]

xu_test = xu[test_time, :]
xs_test = counts[:, test_time].conj().T
xs_test = xs_test[0,:,:]
xs_test = xs_test[:, vocab]


train_authors = (sum(xu_train) > 0)
xu_train = xu_train[:, train_authors]
xu_test = xu_test[:, train_authors]

xs_train, xs_test = tfidf(xs_train, xs_test)
# print(vocab)

k = 500
alpha = 0.5
lambdaa = 0.5
epsilon = 0.001
maxiter = 10
verbose = True
beta = 0.05

lul = pd.read_csv("../LCE_code/LCE_data/AAAAAAAAAAAAAA.csv")
lul.loc[len(lul)] = 0
lul = lul.shift()
lul.loc[0] = 0
lul.columns = range(lul.shape[1])
lul.to_numpy()
# lul.append(pd.Series(name='0'), ignore_index=True)
# lul.index = lul.index + 1
# aaaa = scipy.sparse.csr_matrix(lul.values).toarray()
# a = construct_A(xs_train, 1, True)
a = construct_A(xs_train, 1, True)
# a = a.toarray()
# np.fill_diagonal(a, 0)
w, hu, hs, objhistory = LCE(xs_train,  L2_norm_row(xu_train), a, k, alpha, beta, lambdaa, epsilon, maxiter, verbose)

# w_test = xs_test / hs
# w_test = np.linalg.solve(xs_test, hs)
w_test = np.dot(xs_test, np.linalg.pinv(hs))
w_test[w_test < 0] = 0
lce_rank = np.dot(w_test, hu)
# lce_res = NDCG(lce_rank, xu_test)
lce_res = sk.metrics.ndcg_score(xu_test, lce_rank)
print(lce_res)

# for i in user_item.columns:
#     # print(i)
#     print(nips12['apapers'][i], "+++++",i)
# #     papers = nips12['apapers'][i]
# user_item
# #     if isinstance(papers, int):
#         # print(papers)
#         user_item[papers][i] = 1
#     else:
#         # print(papers)
#         for x in papers:
#             # print(x)
#             user_item[x][i] = 1
# # yes = scipy.sparse.csr_matrix(user_item.values)
# print(user_item.at[308,2027])

# for i in range(user_item.shape[0]):
#     for j in range(user_item.shape[1]):
#         if user_item.at[i,j] == 1:
#             print(i, j)
    # print(col)
    # print(nips12['apapers'][i])

# print(len(nips12['apapers']))

# for i in range(na):
#     # if type(nips12['apapers'][i])  list:
#     if isinstance(nips12['apapers'][i], int):
#         item = nips12['apapers'][i]
#         # print(item)
#         user_item[item][i] = 1
#     else:
#         # print(nips12['apapers'][i])
#         for j in nips12['apapers'][i]:
#             user_item[j][i] = 1

# print(user_item)
#     print(x)
# for x in range(na):
#     print(nips12.get())
    # print(nips12.get(x))

# xu = scipy.sparse.csr_matrix((nd, na), dtype=np.float32)
# # print()
# print(xu)
# # print(len(nips12['apapers']))
# # for i in nips12['apapers']:
# #     if isinstance(i, )
# #     print(i)
# for i in range(na):
#     # print(i)
#     # print(nips12['apapers'][i], "+++++",i)
#     papers = nips12['apapers'][i]
#     if isinstance(papers, int):
#         # print(papers, i)
#         xu[papers, i] = 1
#     else:
#         for x in papers:
#             print(i)
#             # print(x)
#             xu[x, i] = 1
    # if isinstance(nips12['apapers'][i], int):
    #     print(nips12['apapers'][i], "++++++++")
    #     xu[nips12['apapers'][i]][i] = 1
    # else:
    #     papers = (nips12['apapers'][i]).tolist()
    #     print(papers)
    #     for x in papers:
    #         xu[x][i] = 1
    # for x in enumerate(nips12['apapers']):
    #     print(x)
    # papers = (nips12['apapers'][i])
    # print(papers)

    # xu[nips12['apapers'][i],i] = 1
# xu.eliminate_zeros()
# print(xu.todense)
# print(xu.nonzero())
#     for j in nips12['apapers'][i]:
#         print(j)
    # # if type(nips12['apapers'][i])  list:
    # if isinstance(nips12['apapers'][i], int):
    #     item = nips12['apapers'][i]
    #     # print(item)
    #     xu[item, i] = 1.0
    # else:
    #     # print(nips12['apapers'][i])
    #     # print("+++++++")
    #     for j in nips12['apapers'][i]:
    #         # print(j)
    #         xu[j, i] = 1.0

# for i in range(xu.shape[1]):
#     for j in range(xu.shape[0]):
#         if xu[i,j] == 1:
#             print(i, j)
# print(nips12['counts'])
# for i in xu:
#     print(i)
# print(xu[1])
# #
# for i in range(na):
#     # print(xu)
#     # print(nips12['apapers'][i], "++++", i)
#     xu[nips12['apapers'][i], i] = 1
    # except:
    #     continue

# print(xu.nonzero())
#     xu[nips12.get(i),i] = 1

# print(xu)
#     xu[nips12{'apapers'}, i] = 1


# nips12.keys()
# data1 = nips12.get()
# nd = coun