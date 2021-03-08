from engine.evaluation import *
from surprise import Reader
from surprise import SVD
from surprise import KNNBasic
from surprise import NormalPredictor
from surprise import Dataset
from engine.recommender import *
from collections import defaultdict
np.random.seed(1)


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


def format_baselines(predictions):
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))
    return user_est_true


def run_KNN(x_train, x_test, k):
    reader = Reader(rating_scale=(1, 5))
    data_train = Dataset.load_from_df(x_train[['userId', 'movieId', 'rating']], reader)
    data_test = Dataset.load_from_df(x_test[['userId', 'movieId', 'rating']], reader)
    data_train = data_train.build_full_trainset()
    data_test = data_test.build_full_trainset()
    data_testset = data_test.build_testset()
    algo = KNNBasic()
    algo.fit(data_train)
    pr = algo.test(data_testset)
    rec = format_baselines(pr)
    precisions, recalls = precision_recall_at_k(rec, k)
    print(f'|KNN : Precision| = {sum(prec for prec in precisions.values()) / len(precisions)}')
    print(f'|KNN : Recall| = {sum(rec for rec in recalls.values()) / len(recalls)}')


def run_SVD(x_train, x_test, k):
    reader = Reader(rating_scale=(1, 5))
    data_train = Dataset.load_from_df(x_train[['userId', 'movieId', 'rating']], reader)
    data_test = Dataset.load_from_df(x_test[['userId', 'movieId', 'rating']], reader)
    data_train = data_train.build_full_trainset()
    data_test = data_test.build_full_trainset()
    data_testset = data_test.build_testset()
    algo = SVD()

    algo.fit(data_train)
    pr = algo.test(data_testset)
    rec = format_baselines(pr)
    precisions, recalls = precision_recall_at_k(rec, k)
    print(f'|SVD : Precision| = {sum(prec for prec in precisions.values()) / len(precisions)}')
    print(f'|SVD : Recall| = {sum(rec for rec in recalls.values()) / len(recalls)}')


def run_NORMPRED(x_train, x_test, k):
    reader = Reader(rating_scale=(1, 5))
    data_train = Dataset.load_from_df(x_train[['userId', 'movieId', 'rating']], reader)
    data_test = Dataset.load_from_df(x_test[['userId', 'movieId', 'rating']], reader)
    data_train = data_train.build_full_trainset()
    data_test = data_test.build_full_trainset()
    data_testset = data_test.build_testset()
    algo = NormalPredictor()

    algo.fit(data_train)
    pr = algo.test(data_testset)
    rec = format_baselines(pr)
    precisions, recalls = precision_recall_at_k(rec, k)
    print(f'|NORMAL PREDICTOR : Precision| = {sum(prec for prec in precisions.values()) / len(precisions)}')
    print(f'|NORMAL PREDICTOR : Recall| = {sum(rec for rec in recalls.values()) / len(recalls)}')

