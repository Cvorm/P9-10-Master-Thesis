import pandas as pd
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise import SVDpp
from surprise import Dataset
from surprise.model_selection import cross_validate
from node2vec import Node2Vec
from engine.recommender import *
from surprise.model_selection import PredefinedKFold
from surprise import accuracy
from collections import defaultdict

# df = pd.read_csv("testtest.csv", low_memory=False)
x_train, x_test = run_data()
print(x_train)

def get_top_n(predictions, n, min_rating):
    topN = defaultdict(list)
    for userID, movieID, actual_rating, estimated_rating, _ in predictions:
        if (estimated_rating >= min_rating):
            topN[userID].append((movieID, estimated_rating))

    for userID, ratings in topN.items():
        ratings.sort(key=lambda x: x[1], reverse=True)
        topN[userID] = ratings[:n]

    return topN

def make_data(data, name):
    files_dir = "C:\\Users\\Darkmaster\\PycharmProjects\\Recommender\\Data\\Cvorm\\"
    data.to_csv(files_dir + name, header=False, index=False)
    # reader = Reader(line_format='user item rating timestamp', sep=',')
    # train_file = files_dir + "training.csv"
    # test_file = files_dir + "testing.csv"
    # print(train_file)
    #
    # folds_files = [(train_file, test_file)]
    # print(folds_files)
    # data = Dataset.load_from_folds(folds_files, reader=reader)
    # pkf = PredefinedKFold()


def hitrate(topNpredictions, leftoutpredictions):
    hits = 0
    total = 0
    for leftout in leftoutpredictions:
        uid = leftout[0]
        leftoutmovieid = leftout[1]
        hit = False
        for movieId, predictedRating in topNpredictions[uid]:
            if movieId == leftoutmovieid:
                hit = True
        if (hit):
            hits += 1
        total += 1

    return hits / total


# def hitrate (top_n_predictions, leftout_predictions):
#     hits = 0
#     total = 0
#     # users = []
#     # for i in top_n_predictions:
#     #     users.append(i)
#     # print(top_n_predictions)
#     for user in top_n_predictions:
#         for movieID, predictedRating in top_n_predictions[user]:
#
#             hit = False
#             for leftout in leftout_predictions:
#                 mid = leftout[0]
#                 if movieID == mid:
#                     hit = True
#             if hit:
#                 hits += 1
#             total += 1
#
#     return hits / total

def run_SVD():
    files_dir = "C:\\Users\\Darkmaster\\PycharmProjects\\Recommender\\Data\\Cvorm\\"
    x_train, x_test = run_data()

    make_data(x_train, "training.csv")
    make_data(x_test, "testing.csv")
    reader = Reader(line_format='user item rating', sep=',')

    train_file = files_dir + "training.csv"
    test_file = files_dir + "testing.csv"
    print(train_file)

    folds_files = [(train_file, test_file)]
    print(folds_files)
    data = Dataset.load_from_folds(folds_files, reader=reader)
    pkf = PredefinedKFold()
    algo = SVD()

    for trainset, testset in pkf.split(data):
        algo.fit(trainset)
        predictions = algo.test(testset)
        topN_pred = get_top_n(predictions, 10, 4.0)
        # print(topN_pred)
        # for i in topN_pred:
        #     print(i)
        #     for j in topN_pred[i]:
        #         print(j)
        myes = hitrate(topN_pred, predictions)
        print(myes)

        # accuracy.rmse(predictions, verbose=True)


    # print(x_train)
    # reader = Reader(rating_scale=(1,5))
    # data_train = Dataset.load_from_df(x_train[['userId', 'movieId', 'rating']], reader)
    # data_test = Dataset.load_from_df(x_test[['userId', 'movieId', 'rating']], reader)
    # # print(df)
    #
    # # Use the famous SVD algorithm.
    # algo = SVD()
    # algo.fit(data_train)
    # algo.predict(data_test)

    # Run 5-fold cross-validation and print results.
    # res = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=cross_val_num, verbose=True)
    # return res

# def run_SVD_PlusPlus(cross_val_num):
#
#     reader = Reader(rating_scale=(1,5))
#     data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
#     print(df)
#
#     # Use the famous SVD algorithm.
#     algo = SVDpp()
#
#     # Run 5-fold cross-validation and print results.
#     res = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=cross_val_num, verbose=True)
#     return res

yes = run_SVD()
# print(yes)
