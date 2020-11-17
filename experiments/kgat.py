import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import numpy as np
import imdb
import csv
import networkx as nx
from networkx import *
def split_data(data):
    df = pd.read_csv(data)
    ranks = df.groupby('userId')['timestamp'].rank(method='first')
    counts = df['userId'].map(df.groupby('userId')['timestamp'].apply(len))
    # myes = (ranks / counts) > 0.8
    df['new_col'] = (ranks / counts) > 0.8
    # print(myes)
    print(df.head())
    train = df.loc[df['new_col'] == False]
    test = df.loc[df['new_col'] == True]

    train = train.drop(['new_col'], axis=1)
    test = test.drop(['new_col'], axis=1)

    train.to_csv(r'C:\Users\Darkmaster\PycharmProjects\Recommender\Data\Cvorm\training.csv', header=False, index=False)
    test.to_csv(r'C:\Users\Darkmaster\PycharmProjects\Recommender\Data\Cvorm\testing.csv', header=False, index=False)

    # print(test.head())


    # ----AND THEN SAVE THOSE AS CSV----

    # for row in df.index
    # print(test_train)
    # print(ranks.head())
    # print(counts.head())



# def make_train_or_test_txt(ratingdata):
#     df = pd.read_csv(ratingdata)
#     users = []
#     [users.append(x) for x in df["userId"] if x not in users]
#     print(users)
#     with open('Data/KGAT/train.txt', 'w') as f:
#         # writer = csv.writer(f, delimiter='\t')
#         for x in users:
#             items = []
#             items = df.query('userId == {}'.format(x))["movieId"]
#             items = items.values.tolist()
#             stringerbell = ''.join((str(e) + "\t") for e in items)
#             print(stringerbell)
#             # writer.writerow("{}{}".format(x, items))
#             # writer.writerow(str(x) + stringerbell)
#             f.write(str(x) + "\t" + stringerbell + "\n")
#             # print(items)
#     # for j in range(len(df)):
#     #     try:
#     #         getitems = [x for x in df.loc[df["movieId"]]]
#     #     except:
#     #         continue
#     print(df.head())





# make_train_or_test_txt('Data/ratings.csv')
split_data('C:\\Users\\Darkmaster\\PycharmProjects\\Recommender\\Data\\ratings.csv')