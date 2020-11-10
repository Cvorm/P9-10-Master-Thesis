import pandas as pd
from surprise import Dataset
from surprise import Reader

df = pd.read_csv("testtest.csv")

data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], Reader)
