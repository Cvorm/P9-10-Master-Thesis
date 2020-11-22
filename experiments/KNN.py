import pandas as pd
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise import SVDpp
from surprise import Dataset
from surprise.model_selection import cross_validate
from node2vec import Node2Vec
from surprise.model_selection import PredefinedKFold

from surprise import KNNBasic
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise import accuracy


files_dir = "C:\\Users\\Darkmaster\\PycharmProjects\\Recommender\\Data\\Cvorm\\"

reader = Reader(line_format='user item rating timestamp', sep=',')
train_file = files_dir + "training.csv"
test_file = files_dir + "testing.csv"
print(train_file)

folds_files = [(train_file, test_file)]
print(folds_files)
data = Dataset.load_from_folds(folds_files, reader=reader)
pkf = PredefinedKFold()

# algo = KNNBasic()
# algo = SVD()
for trainset, testset in pkf.split(data):

    algo.fit(trainset)
    predictions = algo.test(testset)

    accuracy.rmse(predictions, verbose=True)

# # train_path = "C:\\Users\\Darkmaster\\PycharmProjects\\Recommender\\Data\\Cvorm\\training.csv"
# # test_path = "C:\\Users\\Darkmaster\\PycharmProjects\\Recommender\\Data\\Cvorm\\testing.csv"
# # data = pd.read_csv("C:\\Users\\Darkmaster\\PycharmProjects\\Recommender\\Data\\ratings.csv")
# df_train = pd.read_csv("C:\\Users\\Darkmaster\\PycharmProjects\\Recommender\\Data\\Cvorm\\training.csv", low_memory=False)
# df_test = pd.read_csv("C:\\Users\\Darkmaster\\PycharmProjects\\Recommender\\Data\\Cvorm\\testing.csv", low_memory=False)
#
# print(df_train)
#
# reader = Reader(rating_scale=(1, 5))
#
# data_train = Dataset.load_from_df(df_train[['userId', 'movieId', 'rating']], reader)
# trainset = data_train.build_full_trainset()
#
# data_test = Dataset.load_from_df(df_test[['userId', 'movieId', 'rating']], reader)
#
#
# # data_yes = Dataset.load_from_df(data[['userId', 'movieId', 'rating']], reader)
# # data_train = Dataset.load_from_file(train_path, reader=reader)
# # data_test = Dataset.load_from_file(test_path, reader=reader)
#
# # print(data_train)
# algo = KNNBasic()
#
# algo.fit(trainset)
# predictions = algo.test(data_test)
# #
# # print(accuracy.rmse(predictions))
#
#
# # reader = Reader(rating_scale=(1, 5))
# # data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
#
# # algo = KNNBasic
# # print(df)
#
# # Use the famous SVD algorithm.
# algo = SVD()