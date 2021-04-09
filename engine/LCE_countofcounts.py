from matrix_fac import *
from LCE_code import *
from LCE_code.construct_A import *
from LCE_code.LCE_Beta0 import *
import pandas as pd
from sklearn.model_selection import KFold

# user_user = pd.read_csv("../engine/user_user_matrix.csv", sep='\t', index_col=0, low_memory=False, dtype=float)
# user_user = pd.read_csv("../engine/user_user_matrix.csv", sep='\t', index_col=0)
item_item = pd.read_csv("../engine/item_item_matrix_peterrrrrrrr.csv", sep='\t', index_col=0)
user_item = pd.read_csv("../engine/user_item_matrix_peter.csv", sep='\t', index_col=0)

# user_item = user_item.values
# user_user = user_user.to_numpy()
# print(user_user)
# item_item = pd.read_csv("../engine/item_item_matrix.csv", sep='\t')
# interaction_matrix = pd.read_csv("../engine/user_item_rating_matrix.csv.csv", sep='\t')

# interaction_matrix = interaction_matrix.to_numpy(dtype=float)
# user_user = user_user.values
# print(user_user)

# kf = KFold(n_splits=5)
# kf.get_n_splits(user_item)
# #
# for training, testing in kf.split(user_item):
#     X_train, X_test = user_item.iloc[training], user_item.iloc[testing]
#     print(X_train, X_test)
#     X_train.to_csv(f'user_item_rating_matrix{len(X_train)}.csv', sep='\t')
#     # print(X_train, X_test)

kf = KFold(n_splits=2)
kf.get_n_splits(item_item)
xu_train_list = []
xu_test_list = []
xi_train_list = []
xi_test_list = []

# print(item_item)
#
# for training, testing in kf.split(item_item):
#     X_train, X_test = item_item.iloc[training], item_item.iloc[testing]
#     print(X_train, X_test)
#     X_train.to_csv(f'item_item_rating_matrix{len(X_train)}.csv', sep='\t')
#     # print(X_train, X_test)

for training, testing in kf.split(item_item):
    X1_train, X1_test = item_item.iloc[training], item_item.iloc[testing]
    xi_train_list.append(X1_train.to_numpy())
    xi_test_list.append(X1_test.to_numpy())

for training, testing in kf.split(user_item):
    X2_train, X2_test = user_item.iloc[training], user_item.iloc[testing]
    xu_train_list.append(X2_train.to_numpy())
    xu_test_list.append(X2_test.to_numpy())

print(len(xu_train_list), len(xu_test_list), len(xi_train_list), len(xi_test_list))

k = 500
alpha = 0.5
lambdaa = 0.5
epsilon = 0.001
maxiter = 50
verbose = True
beta = 0.0
# beta = 0.05

for xu_train, xu_test, xi_train, xi_test in zip(xu_train_list, xu_test_list, xi_train_list, xi_test_list):

    a = construct_A(xi_train, 1, True)
    w, hu, hs, objhistory = LCE(xi_train, L2_norm_row(xu_train), a, k, alpha, beta, lambdaa, epsilon, maxiter, verbose)
    # print(type(X_train))
    # print(X_train)
    # print(X_test)
    # print(X_train.to_numpy())
    # xi_train.append(X_train.to_numpy())

    # print(X_train, X_test)
    # X_train.to_csv(f'item_item_rating_matrix{len(X_train)}.csv', sep='\t')
    # print(X_train, X_test)

# print(xi_train)