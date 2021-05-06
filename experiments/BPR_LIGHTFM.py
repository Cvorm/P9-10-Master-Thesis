import numpy as np
import pandas as pd
import scipy.sparse
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, recall_at_k


def run_BPR(split, user_item):
    # train_sparse = scipy.sparse.csr_matrix(train_splt.values)
    # test_sparse = scipy.sparse.csr_matrix(test_split.values)
    train = user_item
    test = user_item * 0

    for index, row in split.iterrows():
        test['u' + str(row['userId'])]['m' + str(row['movieId'])] = row['rating']
        train['u' + str(row['userId'])]['m' + str(row['movieId'])] = 0


    train_sparse = scipy.sparse.csr_matrix(train.values)
    test_sparse = scipy.sparse.csr_matrix(test.values)

    model = LightFM(loss='bpr')
    model.fit(train_sparse, epochs=30, num_threads=4)

    print("####################################################################################")
    print("Test precision: %.2f" % precision_at_k(model, test_sparse, k=10).mean())
    print("Test recall: %.2f" % recall_at_k(model, test_sparse, k=10).mean())
    print("####################################################################################")