import pandas as pd
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise import SVDpp
from surprise import Dataset
from surprise.model_selection import cross_validate
from node2vec import Node2Vec

df = pd.read_csv("testtest.csv", low_memory=False)


def run_SVD(cross_val_num):

    reader = Reader(rating_scale=(1,5))
    data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
    print(df)

    # Use the famous SVD algorithm.
    algo = SVD()

    # Run 5-fold cross-validation and print results.
    res = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=cross_val_num, verbose=True)
    return res

def run_SVD_PlusPlus(cross_val_num):

    reader = Reader(rating_scale=(1,5))
    data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
    print(df)

    # Use the famous SVD algorithm.
    algo = SVDpp()

    # Run 5-fold cross-validation and print results.
    res = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=cross_val_num, verbose=True)
    return res

yes = run_SVD_PlusPlus(5)
print(yes)

def run_node_2_vec():
    