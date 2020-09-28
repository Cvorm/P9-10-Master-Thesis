import numpy as np
import pandas as pd

import networkx as nx
from networkx import *

import surprise as sur
from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import GridSearchCV
from surprise import accuracy
from surprise.model_selection import train_test_split, cross_validate, KFold

data = pd.read_csv('Data/ratings.csv')

data['userId'] = 'u' + data['userId'].astype(str)
data['movieId'] = 'i' + data['movieId'].astype(str)

print(data)

G = nx.Graph()

G.add_nodes_from(data.userId, bipartite=0)
G.add_nodes_from(data.movieId, bipartite=1)

G.add_weighted_edges_from([(uId, mId,rating) for (uId, mId, rating)
              in data[['userId', 'movieId', 'rating']].to_numpy()])

print(info(G))

print(G.is_directed(), G.is_multigraph(), is_bipartite(G))
pagerank = nx.pagerank(G, alpha=0.9)