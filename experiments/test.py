import numpy as np
import pandas as pd
import imdb
import csv

moviesDB = imdb.IMDb()

import surprise as sur
from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import GridSearchCV
from surprise import accuracy
from surprise.model_selection import train_test_split, cross_validate, KFold

data = pd.read_csv('../Data/ratings.csv')
# print(data.head(10))
# print(data)
# data = pd.Dataset.load_builtin("ml-100k")
reader = Reader(rating_scale=(1, 10))
data_test = Dataset.load_from_df(data[['userId', 'movieId', 'rating']], reader)

KG = pd.read_csv('../Data/knowledge-tree.csv', delimiter='\t', encoding='utf-8')
print(KG.head(10))
print(data.head(10))
# testdata = Dataset.load_from_df(data[[]], reader)
# print(data)
# param_grid = {
#      "n_epochs": [5, 10],
#      "lr_all": [0.002, 0.005],
#      "reg_all": [0.4, 0.6]
# }
# gs = GridSearchCV(SVD, param_grid, measures=["rmse", "mae"], cv=3)
#
# gs.fit(testdata)
#
# print(gs.best_score["rmse"])
# print(gs.best_params["rmse"])

# trainset, testset = train_test_split(data_test, test_size=0.25)
# algo = SVD()
#
# algo.fit(trainset)
# predictions = algo.test(testset)
#
# accuracy.rmse(predictions)

# algo = SVD()
# cross_validate(algo, data_test, measures=['RMSE', 'MAE'], cv=5, verbose=True)


# kf = KFold(n_splits=3)
#
# algo = SVD()
#
# for trainset, testset in kf.split(KG):
#
#     algo.fit(trainset)
#     predictions = algo.test(testset)
#
#     accuracy.rmse(predictions, verbose=True)