from matrix_fac import *
from LCE_code import *
from LCE_code.construct_A import *
from LCE_code.LCE_Beta0 import *
from evaluation import *
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import sklearn as sk

# user_user = pd.read_csv("../engine/user_user_matrix.csv", sep='\t', index_col=0, low_memory=False, dtype=float)
# user_user = pd.read_csv("../engine/user_user_matrix.csv", sep='\t', index_col=0)
item_item = pd.read_csv("../engine/item_item_matrix_peterrrrrrrr_correct_mirrored.csv", sep='\t', index_col=0, low_memory=False)
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
def get_rows_cols_numpy_from_df(df):
    cols = df.columns.values.tolist()
    rows = list(df.index)
    numpy = df.to_numpy()
    return rows, cols, numpy

def rows_cols_numpy_to_df(rows, cols, numpy):
    df = pd.DataFrame(data=numpy, index=rows, columns=cols)
    return df
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
# "########################## CROSS-VALIDATION ##########################"
# for training, testing in kf.split(item_item):
#     X1_train, X1_test = item_item.iloc[training], item_item.iloc[testing]
#     xi_train_list.append(X1_train.to_numpy())
#     xi_test_list.append(X1_test.to_numpy())
#
# for training, testing in kf.split(user_item):
#     X2_train, X2_test = user_item.iloc[training], user_item.iloc[testing]
#     xu_train_list.append(X2_train.to_numpy())
#     xu_test_list.append(X2_test.to_numpy())
# "######################################################################"

"########################## TRAIN-TEST-SPLIT ##########################"
X1_train, X1_test = train_test_split(item_item, test_size=0.2, random_state=1)
X2_train, X2_test = train_test_split(user_item, test_size=0.2, random_state=1)

rows1, cols1, numpy1 = get_rows_cols_numpy_from_df(X1_train)
rows2, cols2, numpy2 = get_rows_cols_numpy_from_df(X2_train)
rows3, cols3, numpy3 = get_rows_cols_numpy_from_df(X1_test)
rows4, cols4, numpy4 = get_rows_cols_numpy_from_df(X2_test)

# print(X1_train, X2_train, X1_test, X2_test)
# exit(0)
#
# X1_train = X1_train.to_numpy()
# X2_train = X2_train.to_numpy()
# X1_test = X1_test.to_numpy()
# X2_test = X2_test.to_numpy()
#
xi_train_list.append(numpy1)
xi_test_list.append(numpy3)
xu_train_list.append(numpy2)
xu_test_list.append(numpy4)

"######################################################################"
print(len(xu_train_list), len(xu_test_list), len(xi_train_list), len(xi_test_list))

prec_rec_at = 10
k = 500
alpha = 0.5
lambdaa = 0.5
epsilon = 0.001
maxiter = 100
verbose = True
beta = 0.0
# beta = 0.05
offset = len(xu_train_list[0][:,0])
iteration = 1

for xu_train, xu_test, xi_train, xi_test in zip(xu_train_list, xu_test_list, xi_train_list, xi_test_list):

    a = construct_A(xi_train, 1, True)
    w, hu, hs, objhistory = LCE(xi_train, L2_norm_row(xu_train), a, k, alpha, beta, lambdaa, epsilon, maxiter, verbose)
    # w, hu, hs, objhistory = LCE(xi_train, xu_train, a, k, alpha, beta, lambdaa, epsilon, maxiter, verbose)

    w_test = np.dot(xi_test, np.linalg.pinv(hs)) #could be wrong: linalg.lstsq(b.T, a.T)[0]
    # w_test = np.linalg.lstsq(xi_test.T, hs.T)[0].T
    w_test[w_test < 0] = 0
    pred = np.dot(w_test, hu)
    pred_list = []
    test_list = []

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
    # # prec = recommender_precision(p)
    # pred_list = []
    # test_list = []
    # for x, y in zip(pred.T, xu_test.T):
    #     movie_indexes = np.argsort(x)
    #     reversed = movie_indexes[::-1]
    #     index_offset = [x + (offset * iteration) for x in reversed]
    #     pred_list.append(index_offset[:prec_rec_at])
    #
    #     myes, = np.where(y > 0)
    #     myess = [x + (offset * iteration) for x in myes]
    #     test_list.append(myess)
    #
    # precision = recommender_precision(pred_list, test_list)
    # recall = recommender_recall(pred_list, test_list)
    # print(precision, recall)
    "######################################################################################################"
    # for x, y in zip(pred, xu_test):
    pred_df = rows_cols_numpy_to_df(rows4, cols4, pred)
    xu_test_df = rows_cols_numpy_to_df(rows4, cols4, xu_test)

    for column in pred_df:
        user = pred_df[column]
        sorted = user.sort_values(ascending=False)
        pred_movies = list(sorted.index)
        pred_list.append(pred_movies[:prec_rec_at])
        # print("xdxd")

    for column in xu_test_df:
        user_test = xu_test_df[column]
        filtered = user_test.where(user_test > 0)
        # true_movies = list(filtered.index)
        true_movies = user_test[user_test > 0]
        true_movie_ids = list(true_movies.index)
        test_list.append(true_movie_ids)

    precision = recommender_precision(pred_list, test_list)
    recall = recommender_recall(pred_list, test_list)
    print(precision, recall)

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