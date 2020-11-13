import pandas as pd
import networkx as nx
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise import SVDpp
from surprise import Dataset
from surprise.model_selection import cross_validate
from node2vec import Node2Vec

import numpy as np
import random
from tqdm import tqdm
import re
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def make_edgelist(filename):
    df = pd.read_csv(filename, names=['head','relation','tail'] ,sep='\t')
    df = df.drop('relation', 1)
    # edgelist = list(zip(df['head'],df['tail']))
    # print(df.head())
    # print(edgelist)
    # return edgelist
    return df



train_edges = make_edgelist("Data/movie-train.txt")
# print(train_edges)
# test_edges = make_edgelist("Data/movie-test.txt")
# validation_edges = make_edgelist("Data/movie-valid.txt")
# G = nx.parse_edgelist(train_edges)
G = nx.from_pandas_edgelist(train_edges, 'head', 'tail', create_using=nx.Graph())
# G2 = nx.parse_edgelist(train_edges['head'].tolist(), train_edges['tail'].tolist())
# print(G.nodes, G.edges)

nodes = train_edges['head'].tolist() + train_edges['tail'].tolist()
nodes = list(dict.fromkeys(nodes))
new = []
for x in nodes:
    new.append(str(x))

print(new)
nodes_sorted = sorted(new)
# print(sorted_list)
# nodes_sorted = sorted(nodes)
# print(nodes_sorted)
adj_G = nx.to_pandas_adjacency(G).reindex(index=nodes_sorted, columns=nodes_sorted)
print(adj_G)
# adj_G = nx.to_numpy_matrix(G, nodelist = nodes)
# nodes_sorted = sorted(G.nodes())

# print(adj_G)
# for i in range(adj_G.shape[0]):
#     print(str(i))
#     for j in range(adj_G.shape[1]):
#         if adj_G[i,j] == 1:
#             print(adj_G[i,j], i ,j)
#
# print(adj_G)
# # print(adj_G.shape)
#
#

# yes = adj_G.columns.values[0]
# yes2 = adj_G.index.values[0]
# print(adj_G.iloc[0][0])
# print(yes, yes2)

all_unconnected_pairs = []
# traverse adjacency matrix
offset = 0
for i in range(adj_G.shape[0]):
  for j in range(offset,adj_G.shape[1]):
    if i != j:
        try:
          # if nx.shortest_path_length(G, str(i), str(j)) <=2:
          if nx.shortest_path_length(G, adj_G.index.values[i], adj_G.columns.values[j]) <= 2:
            if adj_G.iloc[i][j] == 0:
              all_unconnected_pairs.append([nodes_sorted[i],nodes_sorted[j]])
        except:
            continue

  offset = offset + 1
print(len(all_unconnected_pairs))
print(all_unconnected_pairs[0])


node_1_unlinked = [i[0] for i in all_unconnected_pairs]
node_2_unlinked = [i[1] for i in all_unconnected_pairs]


data = pd.DataFrame({'head':node_1_unlinked, 'tail':node_2_unlinked})

data['link'] = 0

# print(data)

initial_node_count = len(G.nodes)

edgelist_temp = train_edges.copy()
complete = nx.from_pandas_edgelist(edgelist_temp, 'head', 'tail', create_using=nx.Graph())
clusters = nx.number_connected_components(complete)
print("++++++++++++++", clusters)
# empty list to store removable links
omissible_links_index = []
print(edgelist_temp)
for i in train_edges.index.values:
    # print(i)
    # remove a node pair and build a new graph
    G_temp = nx.from_pandas_edgelist(edgelist_temp.drop(index=i), "head", "tail", create_using=nx.Graph())
    # nx.number_connected_components(G_temp) ==
    # check there is no spliting of graph and number of nodes is same
    if (nx.number_connected_components(G_temp) == clusters) and (len(G_temp.nodes) == initial_node_count):
        # print("doooooooood")
        omissible_links_index.append(i)
        edgelist_temp = edgelist_temp.drop(index=i)


# create dataframe of removable edges

print(omissible_links_index)
edgelist_ghost = train_edges.loc[omissible_links_index]
# print(edgelist_ghost)
# # add the target variable 'link'
edgelist_ghost['link'] = 1
data = data.append(edgelist_ghost[['head', 'tail', 'link']], ignore_index=True)
print(data['link'].value_counts())
# # print(nodes)
# print(data['link'].value_counts())
edgelist_partial = train_edges.drop(index=edgelist_ghost.index.values)
G_data = nx.from_pandas_edgelist(edgelist_partial, "head", "tail", create_using=nx.Graph())

# Generate walks
node2vec = Node2Vec(G_data, dimensions=100, walk_length=16, num_walks=50)

# train node2vec model
n2w_model = node2vec.fit(window=7, min_count=1)

x = [(n2w_model[str(i)]+n2w_model[str(j)]) for i,j in zip(data['head'], data['tail'])]
xtrain, xtest, ytrain, ytest = train_test_split(np.array(x), data['link'],
                                                test_size = 0.3,
                                                random_state = 35)

lr = LogisticRegression(class_weight="balanced", max_iter=5000)

lr.fit(xtrain, ytrain)
predictions = lr.predict_proba(xtest)
print(roc_auc_score(ytest, predictions[:,1]))
# edges = make_edgelist("Data/movie-train.txt")
# graph = nx.Graph(edges)
# node2vec_obj = Node2Vec(graph, dimensions=20, walk_length=30, num_walks=200, p=1, q=1, workers=1)
# node2vec_model = node2vec_obj.fit(window=10, min_count=1, batch_words=4)
# print(node2vec_model.wv.most_similar('m129'))