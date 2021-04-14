from matrix_fac import *
from LCE_code import *
from LCE_code.construct_A import *
from LCE_code.LCE_Beta0 import *
from evaluation import *
import pandas as pd
from sklearn.model_selection import KFold
import sklearn as sk

# user_user = pd.read_csv("../engine/user_user_matrix.csv", sep='\t', index_col=0, low_memory=False, dtype=float)
# user_user = pd.read_csv("../engine/user_user_matrix.csv", sep='\t', index_col=0)
item_item = pd.read_csv("../engine/item_item_matrix_peterrrrrrrr.csv", sep='\t', index_col=0, low_memory=False)
# user_item = pd.read_csv("../engine/user_item_matrix_peter.csv", sep='\t', index_col=0)
# user_item = pd.read_csv("../engine/user_item_matrix_peterr_ratings.csv", sep='\t', index_col=0)
user_item = pd.read_csv("../engine/user_item_ny.csv", sep='\t', index_col=0, low_memory=False)

print(item_item.shape, user_item.shape)
# exit(0)
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
res = []
res_apk = []

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
maxiter = 10
verbose = True
beta = 0.0
# beta = 0.05
offset = len(xu_train_list[0][:,0])
iteration = 0

for xu_train, xu_test, xi_train, xi_test in zip(xu_train_list, xu_test_list, xi_train_list, xi_test_list):

    a = construct_A(xi_train, 1, True)
    w, hu, hs, objhistory = LCE(xi_train, L2_norm_row(xu_train), a, k, alpha, beta, lambdaa, epsilon, maxiter, verbose)

    w_test = np.dot(xi_test, np.linalg.pinv(hs)) #PROBABLY WRONG!!!! linalg.lstsq(b.T, a.T)[0]
    w_test[w_test < 0] = 0
    pred = np.dot(w_test, hu)

    # "#########################################-APK-#########################################"
    # temp = []
    # for x, y in zip(pred.T, xu_test.T):
    #
    #     movie_indexes = np.argsort(x)
    #     reversed = movie_indexes[::-1]
    #     index_offset = [x + (offset*iteration) for x in reversed]
    #
    #     myes, = np.where(y == 1)
    #     myess = [x + (offset*iteration) for x in myes]
    #     temp.append(apk(myess, index_offset, 5))
    # res_apk.append((sum(temp)/len(temp)))
    # "#####################################################################################"
    "######################################### Precision & Recall #########################################"
    # prec = recommender_precision(p)
    pred_list = []
    test_list = []
    for x, y in zip(pred.T, xu_test.T):
        movie_indexes = np.argsort(x)
        reversed = movie_indexes[::-1]
        index_offset = [x + (offset * iteration) for x in reversed]
        pred_list.append(index_offset)

        myes, = np.where(y == 1)
        myess = [x + (offset*iteration) for x in myes]
        test_list.append(y)

    precision = recommender_precision(pred_list, test_list)
    recall = recommender_recall(pred_list, test_list)
    print(precision, recall)
    "######################################################################################################"


    # myes = np.argsort(pred[:,0])
    # myes_reverse = myes[::-1]
    # myesss = [x + (offset*iteration) for x in myes_reverse]
    # yeshu = xu_test[:,0]
    # # yeshhhuu = [x + (offset*iteration) for x in yeshu if x == 1]
    # yessshuu, = np.where(yeshu == 1)
    # yeshhhhhhhh = [x + (offset*iteration) for x in yessshuu]
    # print(type(myesss), type(yeshhhhhhhh))
    # pls = apk(yeshhhhhhhh, myesss, 5)
    # np.where()
    # res.append(sk.metrics.ndcg_score(xu_test, pred))
    # res.append(sk.metrics.average_precision_score(xu_test, pred))
    # print(res)
    # print("myes")
    iteration += 1
# print(sum(res) / len(res))
print(res_apk)
    # print(type(X_train))
    # print(X_train)
    # print(X_test)
    # print(X_train.to_numpy())
    # xi_train.append(X_train.to_numpy())

    # print(X_train, X_test)
    # X_train.to_csv(f'item_item_rating_matrix{len(X_train)}.csv', sep='\t')
    # print(X_train, X_test)

# print(xi_train)