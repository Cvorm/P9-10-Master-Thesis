from engine.matrix_fac import *
from engine.LCE_code import *
from engine.LCE_code.construct_A import *
from engine.LCE_code.LCE_Beta0 import *
from engine.evaluation import *

from engine.run import run_mml

from experiments.LIGHTFM import *
import pandas as pd
from sklearn.model_selection import KFold
import statistics
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn as sk
import subprocess
import sys
from experiments.BPR_LIGHTFM import *

# user_user = pd.read_csv("../engine/user_user_matrix.csv", sep='\t', index_col=0, low_memory=False, dtype=float)
# user_user = pd.read_csv("../engine/user_user_matrix.csv", sep='\t', index_col=0)
# item_item = pd.read_csv("../engine/item_item_matrix_peterrrrrrrr_correct_mirrored.csv", sep='\t', index_col=0, low_memory=False)
#item_item = pd.read_csv("../engine/item_feature_matrix.csv", sep='\t', index_col=0, low_memory=False)
# user_item = pd.read_csv("../engine/user_item_matrix_peter.csv", sep='\t', index_col=0)
# user_item = pd.read_csv("../engine/user_item_matrix_peterr_ratings.csv", sep='\t', index_col=0)
# user_item = pd.read_csv("../engine/user_item_ny.csv", sep='\t', index_col=0, low_memory=False)


item_item = pd.read_csv("../engine/item_item_similarity_1000209_ratings.csv", sep='\t', index_col=0, low_memory=False)
#item_item = pd.read_csv("../engine/item_feature_matrix.csv", sep='\t', index_col=0, low_memory=False)
# user_item = pd.read_csv("../engine/user_item_matrix_peter.csv", sep='\t', index_col=0)
# user_item = pd.read_csv("../engine/user_item_matrix_peterr_ratings.csv", sep='\t', index_col=0)
user_item = pd.read_csv("../engine/user_user_matrix_TETETETTETEETET.csv", sep='\t', index_col=0, low_memory=False)
print(item_item.shape, user_item.shape)


list_of_items = [x[0] for x in item_item.iterrows()]
# print(item_feature)

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


kf = KFold(n_splits=10)
kf.get_n_splits(item_item)
# kf.get_n_splits(item_feature) #item_feature
xu_train_list = []
xu_test_list = []
xi_train_list = []
xi_test_list = []
res = []
res_apk = []

xu_train_list_kf = []
xu_test_list_kf = []
xi_train_list_kf = []
xi_test_list_kf = []

train_df_list = []
prec_rec_at = 5


# print(item_item)
#
# for training, testing in kf.split(item_item):
#     X_train, X_test = item_item.iloc[training], item_item.iloc[testing]
#     print(X_train, X_test)
#     X_train.to_csv(f'item_item_rating_matrix{len(X_train)}.csv', sep='\t')
#     # print(X_train, X_test)

    # if == True & int(y[1:]) in split.movieId == True:
    #     print(x, y)
    # if split['movieId'] == x and split['userId'] == y:
    #     return 1
    # else:
    #     return 0
"########################## CROSS-VALIDATION ##########################"
def cross_validation(user, item):
    items = list(np.unique(data['movieId']))
    print(items)
    item_interaction_count = dict.fromkeys(items, 0)

    for idx, m in movieratings.iterrows():
        mid = 'm' + m['movieId'].astype(str)
        item_interaction_count[mid] += 1
    # for k, v in list(item_interaction_count.items()): # code for deleting items with no ratings from dictionary
    #     if v == 0:
    #         del item_interaction_count[k]
    item_interaction_count = dict(sorted(item_interaction_count.items(), key=lambda item: item[1], reverse=True))
    tmp_count = 0
    k_folds = 5
    item_split = [[] for i in range(k_folds)]
    for m in item_interaction_count.keys():
        if tmp_count == k_folds:
            tmp_count = 0
        item_split[tmp_count].append(m)
        tmp_count += 1
    item_df_split = []
    for split in item_split:
        tmp = [int(z[1:]) for z in split]
        test_df = movieratings[movieratings['movieId'].isin(tmp)]
        item_df_split.append(test_df)
    item_df_split_split = []
    # for d in item_df_split: # for splitting the item dataset 50% on ratings
    #     tmp_d = d.sample(frac=1).reset_index(drop=True)
    #     tmp_d = tmp_d.head(round(len(tmp_d) / 2 ))
    #     item_df_split_split.append(tmp_d)
    tmp_count = 0
    for d in item_df_split:
        d = d.sample(frac=1).reset_index(drop=True)
        tmp = pd.DataFrame(columns=['userId', 'movieId', 'rating', 'timestamp'])
        for s in item_split[tmp_count]:
            s = int(s[1:])
            y = d[d['movieId'] == s]
            y = y.head(round(len(y) / 2))
            tmp = tmp.append(y, ignore_index=True)
        tmp_count += 1
        item_df_split_split.append(tmp)

    bpr_prec_list = []
    bpr_rec_list = []
    lightfm_prec_list = []
    lightfm_rec_list = []
    lightfm_nov_list = []

    for split in item_df_split:
        # training = user_item.apply(lambda x: pd.DataFrame(x).apply(lambda y: bitch(x.name, y.name, split)), axis=1)
        # for columns in user_item:100
        #     for y in user_item.itercols():
        #         print(x, y)
        mid_rows = ['m' + str(x) for x in split['movieId'].unique().tolist()]
        train_user_item = user_item.drop([x for x in mid_rows])
        train_rows = list(train_user_item.index)
        test_user_item = user_item.drop([x for x in train_rows])
        rows_train, cols_train, numpy_train = get_rows_cols_numpy_from_df(train_user_item)
        rows_test, cols_test, numpy_test = get_rows_cols_numpy_from_df(test_user_item)
        xu_train_list_kf.append((rows_train, cols_train, numpy_train))
        xu_test_list_kf.append((rows_test, cols_test, numpy_test))

        train_item_item = item_item.drop([x for x in mid_rows])
        test_item_item = item_item.drop([x for x in train_rows])

        rows_train2, cols_train2, numpy_train2 = get_rows_cols_numpy_from_df(train_item_item)
        rows_test2, cols_test2, numpy_test2 = get_rows_cols_numpy_from_df(test_item_item)
        xi_train_list_kf.append((rows_train2, cols_train2, numpy_train2))
        xi_test_list_kf.append((rows_test2, cols_test2, numpy_test2))


        # bpr_prec, bpr_rec = run_BPR(split, user_item, prec_rec_at)
        # bpr_prec_list.append(bpr_prec)
        # bpr_rec_list.append(bpr_rec)

        # exit(0)
        # train = user_item



        data['movieId'] = data['movieId'].str[1:]
        b = movieratings.copy()
        a = movieratings.copy()
        # LightFM
        tmp = [int(z[1:]) for z in rows_test]

        b.loc[b.movieId.isin(tmp), "rating"] = 0
        a.loc[~a.movieId.isin(tmp), "rating"] = 0
        # movieidlist_test = [x for x in b['movieId']]
        # movieidlist_test = list(map(str, movieidlist_test))
        # mov_test = data[data.movieId.isin(movieidlist_test)]

        movieidlist = [x for x in b['movieId']]

        #train_df = movieratings[movieratings['movieId'].isin(tmp)]
        # test_df = movieratings[~movieratings['movieId'].isin(tmp)]     movieidlist = [x for x in b['movieId']]
        test_df = movieratings[movieratings['movieId'].isin(tmp)]
        train_df = movieratings[~movieratings['movieId'].isin(tmp)]
        train_df_list.append(train_df)

        [movieidlist.append(x) for x in a['movieId']]
        mylist = list(dict.fromkeys(movieidlist))
        mylist = list(map(str, mylist))
        tmp_mov = data[data.movieId.isin(mylist)]
        #  vigtigt at movieId er samme TYPE i begge dataframes

        lightfm_prec, lightfm_rec = run_lightfm(tmp_mov, movieratings, b, a, prec_rec_at, train_df)
        run_mml(train_df, test_df, prec_rec_at)
    
        lightfm_prec_list.append(lightfm_prec)
        lightfm_rec_list.append(lightfm_rec)
        # lightfm_nov_list.append(lightfm_nov)

    file_object = open('LIGHTFM_Daniels_BPR.txt', 'a')
    file_object.write(
                      f'prec_rec_at = {prec_rec_at},'
                      # f'avg. novelty = {sum(lightfm_nov_list) / len(lightfm_nov_list)},'
                      f'avg. precision = {sum(lightfm_prec_list) / len(lightfm_prec_list)},'
                      f'avg. recall = {sum(lightfm_rec_list) / len(lightfm_rec_list)}\n')

    # file_object = open('LIGHTFM_BPR.txt', 'a')
    # file_object.write(
    #                   f'prec_rec_at = {prec_rec_at},'
    #                   f'avg. precision = {sum(bpr_prec_list) / len(bpr_prec_list)},'
    #                   f'avg. recall = {sum(bpr_rec_list) / len(bpr_rec_list)}\n')

    # for training, testing in kf.split(item):
    #     X1_train, X1_test = item.iloc[training], item.iloc[testing]
    #     rows_train, cols_train, numpy_train = get_rows_cols_numpy_from_df(X1_train)
    #     rows_test, cols_test, numpy_test = get_rows_cols_numpy_from_df(X1_test)
    #     xi_train_list_kf.append((rows_train, cols_train, numpy_train))
    #     xi_test_list_kf.append((rows_test, cols_test, numpy_test))


    # for training, testing in kf.split(user):
    #     X2_train, X2_test = user.iloc[training], user.iloc[testing]
    #     rows_train, cols_train, numpy_train = get_rows_cols_numpy_from_df(X2_train)
    #     rows_test, cols_test, numpy_test = get_rows_cols_numpy_from_df(X2_test)
    #     xu_train_list_kf.append((rows_train, cols_train, numpy_train))
    #     xu_test_list_kf.append((rows_test, cols_test, numpy_test))


cross_validation(user_item, item_item)
#     xi_train_list.append(X1_train.to_numpy())
#     xi_test_list.append(X1_test.to_numpy())
#
# for training, testing in kf.split(user_item):
#     X2_train, X2_test = user_item.iloc[training], user_item.iloc[testing]
#     xu_train_list.append(X2_train.to_numpy())
#     xu_test_list.append(X2_test.to_numpy())
# "######################################################################"

# "########################## TRAIN-TEST-SPLIT ##########################"
# X1_train, X1_test = train_test_split(item_item, test_size=0.2, random_state=1)
# X2_train, X2_test = train_test_split(user_item, test_size=0.2, random_state=1)
#
# rows1, cols1, numpy1 = get_rows_cols_numpy_from_df(X1_train)
# rows2, cols2, numpy2 = get_rows_cols_numpy_from_df(X2_train)
# rows3, cols3, numpy3 = get_rows_cols_numpy_from_df(X1_test)
# rows4, cols4, numpy4 = get_rows_cols_numpy_from_df(X2_test)
#
# # print(X1_train, X2_train, X1_test, X2_test)
# # exit(0)
# #
# # X1_train = X1_train.to_numpy()
# # X2_train = X2_train.to_numpy()
# # X1_test = X1_test.to_numpy()
# # X2_test = X2_test.to_numpy()
# #
# xi_train_list.append(numpy1)
# xi_test_list.append(numpy3)
# xu_train_list.append(numpy2)
# xu_test_list.append(numpy4)
#
# "######################################################################"
# print(len(xu_train_list), len(xu_test_list), len(xi_train_list), len(xi_test_list))


# k = 100
# alpha = 0.8
# lambdaa = 1
# epsilon = 0.001
# maxiter = 200
# verbose = True
# beta = 0.0 #graph reguralization
# # beta = 0.05
# # offset = len(xu_train_list[0][:,0])
# # iteration = 1
# prec_list = []
# rec_list = []
# "######################################### CROSS VALIDATION KFOLDS #########################################"
# print(len(xu_train_list_kf), len(xu_test_list_kf), len(xi_train_list_kf), len(xi_test_list_kf), len(train_df_list))
#
# for xu_train, xu_test, xi_train, xi_test in zip(xu_train_list_kf, xu_test_list_kf, xi_train_list_kf, xi_test_list_kf):
#     print("yes")
#     a = construct_A(xi_train[2], 1, True)
#     w, hu, hs, objhistory = LCE(xi_train[2], L2_norm_row(xu_train[2]), a, k, alpha, beta, lambdaa, epsilon, maxiter, verbose)
#
#     w_test = np.dot(xi_test[2], np.linalg.pinv(hs)) #could be wrong: linalg.lstsq(b.T, a.T)[0]
#         # w_test = np.linalg.lstsq(xi_test.T, hs.T)[0].T
#     w_test[w_test < 0] = 0
#     pred = np.dot(w_test, hu)
#     pred_list = []
#     nov_list = []
#     test_list = []
#     missing = []
#     pred_dict = defaultdict(list)
#     pred_df = rows_cols_numpy_to_df(xu_test[0], xu_test[1], pred)
#     xu_test_df = rows_cols_numpy_to_df(xu_test[0], xu_test[1], xu_test[2])
#     rating_df = rows_cols_numpy_to_df(xu_train[0], xu_train[1], xu_train[2])
#     item_df = rows_cols_numpy_to_df(xi_train[0], xi_train[1], xi_train[2])
#     for column in pred_df:
#         user = pred_df[column]
#         sorted = user.sort_values(ascending=False)
#         pred_movies = list(sorted.index)
#         pred_list.append(pred_movies[:prec_rec_at])
#         pred_dict[column] = pred_movies[:prec_rec_at]
#     #         # print("xdxd")
#     #
#     for column in xu_test_df:
#         user_test = xu_test_df[column]
#         filtered = user_test.where(user_test > 0)
#         # true_movies = list(filtered.index)
#         true_movies = user_test[user_test > 0]
#         if len(true_movies) > 0:
#             true_movie_ids = list(true_movies.index)
#             test_list.append(true_movie_ids)
#         else:
#             missing.append(int(column[1:]))
#
#     print(missing)
#     for m in missing:
#         del pred_list[m-1]
#     precision = recommender_precision(pred_list, test_list)
#     recall = recommender_recall(pred_list, test_list)
#     users = rating_df.columns.tolist()
#     # nov = novelty(pred_dict, rating_df, list_of_items, users, prec_rec_at)
#     print(precision, recall)
#     # nov_list.append(nov)
#     prec_list.append(precision)
#     rec_list.append(recall)
# # print(f'average novelty: {sum(nov_list) / len(nov_list)}')
# print("average precision:", sum(prec_list) / len(prec_list))
# print("average recall", sum(rec_list) / len(rec_list))
#
# file_object = open('LCE_COUNT.txt', 'a')
# file_object.write(f'settings: k = {k},'
#                   f' alpha = {alpha}, '
#                   f'lambda = {lambdaa}, '
#                   f'epsilon = {epsilon}, '
#                   f'maxiter = {maxiter}, '
#                   f'prec_rec_at = {prec_rec_at},'
#                   # f'avg. novelty = {sum(nov_list) / len(nov_list)},'
#                   f'avg. precision = {sum(prec_list) / len(prec_list)},'
#                   f'avg. recall = {sum(rec_list) / len(rec_list)}\n')
# "############################################################################################################"










# "######################################### TRAIN & TEST SPLIT #########################################"
# for xu_train, xu_test, xi_train, xi_test in zip(xu_train_list, xu_test_list, xi_train_list, xi_test_list):
#
#     a = construct_A(xi_train, 1, True)
#     w, hu, hs, objhistory = LCE(xi_train, L2_norm_row(xu_train), a, k, alpha, beta, lambdaa, epsilon, maxiter, verbose)
#     # w, hu, hs, objhistory = LCE(xi_train, xu_train, a, k, alpha, beta, lambdaa, epsilon, maxiter, verbose)
#
#     w_test = np.dot(xi_test, np.linalg.pinv(hs)) #could be wrong: linalg.lstsq(b.T, a.T)[0]
#     # w_test = np.linalg.lstsq(xi_test.T, hs.T)[0].T
#     w_test[w_test < 0] = 0
#     pred = np.dot(w_test, hu)
#     pred_list = []
#     test_list = []
#
#     # "#########################################-APK-#########################################"
#     # temp = []
#     # for x, y in zip(pred.T, xu_test.T):
#     #
#     #     movie_indexes = np.argsort(x)
#     #     reversed = movie_indexes[::-1]
#     #     index_offset = [x + (offset*iteration) for x in reversed]
#     #
#     #     myes, = np.where(y == 1)
#     #     myess = [x + (offset*iteration) for x in myes]
#     #     temp.append(apk(myess, index_offset, 5))
#     # res_apk.append((sum(temp)/len(temp)))
#     # "#####################################################################################"
#     "######################################### Precision & Recall #########################################"
#     # for x, y in zip(pred, xu_test):
#     pred_df = rows_cols_numpy_to_df(rows4, cols4, pred)
#     xu_test_df = rows_cols_numpy_to_df(rows4, cols4, xu_test)
#
#     for column in pred_df:
#         user = pred_df[column]
#         sorted = user.sort_values(ascending=False)
#         pred_movies = list(sorted.index)
#         pred_list.append(pred_movies[:prec_rec_at])
#         # print("xdxd")
#
#     for column in xu_test_df:
#         user_test = xu_test_df[column]
#         filtered = user_test.where(user_test > 0)
#         # true_movies = list(filtered.index)
#         true_movies = user_test[user_test > 0]
#         true_movie_ids = list(true_movies.index)
#         test_list.append(true_movie_ids)
#
#     precision = recommender_precision(pred_list, test_list)
#     recall = recommender_recall(pred_list, test_list)
#     print(precision, recall)
# "######################################################################################################"

# print(res_apk)
