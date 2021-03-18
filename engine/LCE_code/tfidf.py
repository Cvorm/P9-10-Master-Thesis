import numpy as np
import math
import pandas as pd
import scipy as sp
import scipy.sparse

# function[X_train, X_test] = tfidf(X_train, X_test)
#
# idf = math.log(size(X_train, 1). / (sum(X_train > 0) + eps));
# IDF = spdiags(idf
# ', 0, size(idf,2), size(idf,2));
# X_train = X_train * IDF;
# X_train = L2_norm_row(X_train);
#
# X_test = X_test * IDF;
# X_test = L2_norm_row(X_test);
#
# function
# Xnorm = L2_norm_row(X)
# Xnorm = spdiags(1. / (sqrt(sum(X. * X, 2)) + eps), 0, size(X, 1), size(X, 1)) * X;
# end
#
# end

def tfidf(x_train, x_test):
    idf = np.divide(x_train.shape[0], (sum(x_train > 0) + np.spacing(1)))
    # IDF = spdiags(idf', 0, size(idf,2), size(idf,2));
    IDF = sp.sparse.spdiags(idf, 0, idf.shape[1], idf.shape[1])
    x_train = x_train * IDF
    x_train = L2_norm_row(x_train)

    x_test = x_test * IDF
    x_test = L2_norm_row(x_test)

    return x_train, x_test

def L2_norm_row(X):
    return sp.sparse.spdiags(1. / (np.sqrt(sum(X * X, 2)) + eps), 0, len(X), len(X)) * X