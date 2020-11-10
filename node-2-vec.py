import pandas as pd
import networkx as nx
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise import SVDpp
from surprise import Dataset
from surprise.model_selection import cross_validate
from node2vec import Node2Vec

def make_edgelist(filename):
    df = pd.read_csv(filename, names=['head','relation','tail'] ,sep='\t')
    df = df.drop('relation', 1)
    edgelist = list(zip(df['head'],df['tail']))
    print(df.head())
    # print(edgelist)
    return edgelist

edges = make_edgelist("Data/movie-train.txt")
graph = nx.Graph(edges)
node2vec_obj = Node2Vec(graph, dimensions=20, walk_length=30, num_walks=200, p=1, q=1, workers=1)
node2vec_model = node2vec_obj.fit(window=10, min_count=1, batch_words=4)
print(node2vec_model.wv.most_similar('m129'))