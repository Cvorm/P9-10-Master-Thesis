import csv
import sys
import pandas as pd
from surprise import Reader
from surprise import SVD
from surprise import KNNBasic
from surprise import NormalPredictor
from surprise import Dataset
from engine.recommender import *
from surprise.model_selection import PredefinedKFold
from surprise import accuracy
from collections import defaultdict

# df = pd.read_csv("testtest.csv", low_memory=False)
# x_train, x_test = run_data()
# print(x_train)
inp = sys.argv
k_movies = inp[1]

def precision_recall_at_k(predictions, k=10, threshold=4):
    """Return precision and recall at k metrics for each user"""

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        # n_rel = sum(true_r for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])
        # n_rel_and_rec_k = sum((true_r  and (est >= threshold))
        #                       for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        # When n_rec_k is 0, Precision is undefined. We here set it to 0.

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

        # Recall@K: Proportion of relevant items that are recommended
        # When n_rel is 0, Recall is undefined. We here set it to 0.

        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    return precisions, recalls

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
    data.to_csv(name, header=False, index=False)
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

    print(total)
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

def run_SVD(k):
    files_dir = "C:\\Users\\Darkmaster\\PycharmProjects\\Recommender\\Data\\Cvorm\\"
    x_train, x_test = run_data()

    make_data(x_train, "training.csv")
    make_data(x_test, "testing.csv")
    reader = Reader(line_format='user item rating', sep=',')

    train_file = "training.csv"
    test_file = "testing.csv"
    print(train_file)

    folds_files = [(train_file, test_file)]
    print(folds_files)
    data = Dataset.load_from_folds(folds_files, reader=reader)
    pkf = PredefinedKFold()
    algo = SVD()
    algo1 = KNNBasic()
    algo2 = NormalPredictor()

    for trainset, testset in pkf.split(data):
        algo.fit(trainset)
        algo1.fit(trainset)
        algo2.fit(trainset)
        predictions = algo.test(testset)
        predictions1 = algo1.test(testset)
        predictions2 = algo2.test(testset)
        precisions, recalls = precision_recall_at_k(predictions, k, threshold=4)
        precisions1, recalls1 = precision_recall_at_k(predictions1, k, threshold=4)
        precisions2, recalls2 = precision_recall_at_k(predictions2, k, threshold=4)

        print(f' PRECISION SVD : {sum(prec for prec in precisions.values()) / len(precisions)}')
        print(f' RECALL SVD : {sum(rec for rec in recalls.values()) / len(recalls)}')
        print(f' PRECISION KNN : {sum(prec for prec in precisions1.values()) / len(precisions1)}')
        print(f' RECALL KNN : {sum(rec for rec in recalls1.values()) / len(recalls1)}')
        print(f' PRECISION RAND : {sum(prec for prec in precisions2.values()) / len(precisions2)}')
        print(f' RECALL RAND : {sum(rec for rec in recalls2.values()) / len(recalls2)}')
        # topN_pred = get_top_n(predictions, 10, 4.0)
        # print(topN_pred)
        # for i in topN_pred:
        #     print(i)
        #     for j in topN_pred[i]:
        #         print(j)
        # myes = hitrate(topN_pred, predictions)
        # print(myes)

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
def make_edgelist(filename):
    df = pd.read_csv(filename, names=['head','relation','tail'] ,sep='\t')
    df = df.drop('relation', 1)
    edges = list(zip(df['head'],df['tail']))
    with open('../Data/Cvorm/edgelist.edg', 'w', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        for index, row in df.iterrows():
            writer.writerow((row['head'], row['tail']))
    # f = open("../Data/Cvorm/edgelist.edg", "w")
    # for e in edges:
    #     print(e)
        # f.writelines(e)
    # edgelist = list(zip(df['head'],df['tail']))
    # print(df.head())
    # print(edgelist)
    # return edgelist
    # return df

# def run_node2vec(edgelist):
    # pec.node2vec(edgelist, '../Data/Cvorm/edgelist_embedding.emb')
    # pecanpy.node2vec()
# def make_edg(df, path):


# def run_node2vec():
    # train_edges = make_edgelist("../Data/Cvorm/edges.csv")

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

run_SVD(k_movies)
# run_node2vec("../Data/Cvorm/edgelist.edg")
# make_edgelist("../Data/Cvorm/edges.csv")
# print(yes)
