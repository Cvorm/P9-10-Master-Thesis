from collections import Counter
from anytree import Node, RenderTree
import networkx as nx
import matplotlib.pyplot as plt
import re
import nltk
from nltk import Tree
import numpy as np

# class TET:
#     root = type(Node)
#     tree = []
#     freevars = []
#     children = []
#
#
#     def __init__(self):
#         self.root = Node("root")
#         self.freevars = []
#         self.children = []
#
#     def countLeaf(self):
#
#
#     def countInternal(self):
#
#
#     def makeTET(self, vars):
#         self.freevars = vars

# def Parse(tokens):
#     buffer = []
#     buffer.append(tokens[0])
#     buffer.append(tokens[1])
#     print(tokens)
#     print(buffer)
#     if buffer[0]
# def TETSpec(tet_spec):
#     nltk_tokens = nltk.word_tokenize(tet_spec)
#     graph = nx.Graph()
#     Parse(nltk_tokens)
#     # if(not(nltk_tokens[0] == "[")):
#     #     print("incorrect tet string")
#     #     exit(0)
#     #
#     # Parse(tet_spec)
#     # while(not(nltk_tokens[0] == "]")):
#     #     token = nltk_tokens[0]
#     #     if(not(token == "[")):
#     #         print("incorrect tet string")
#     #         exit(0)
#     #     print(nltk_tokens)
#     #     while()
#     #     nltk_tokens.pop(0)

# def TETSpec(tet_spec):
#
#     tree = Tree.fromstring(tet_spec)
#     tree.pretty_print()
#     # yes = tree.label()
#     # tree.treepositions()
#     graph = nx.Graph()
#     graph.add_node(tree.label())
#     traverse_tree(tree, graph)
#     for x in tree:
# #         print(x)
# def get_count(graph):
#     if descendants()

def traverse_tree(tree):

    print("tree:", tree)
    for subtree in tree:
        if type(subtree) == nltk.tree.Tree:
            traverse_tree(subtree)
    # # print(yes)

# TETSpec("(movie[M](comedy[M])(romance[M]))")
tree = Tree.fromstring("(user(movie(genre)(actor))(movie(genre)(actor)))")
print(tree)
traverse_tree(tree)
    # for token in nltk_tokens:


# def TETspec(tet_spec):
#     tet_tree = nx.Graph()
#     tokens = []
#     nodes = []
#     edges = []
#     if len(tet_spec.split(" | ")) <= 2:
#         print("dude")
#         exit(0)
#     else:
#         tokens = tet_spec.split(" | ")
#         # for x in range(len(tokens)):
#         for x in tokens:
#             if re.search("(-(\([A-Z]*\))->)", x):
#                 edges.append(x)
#             elif re.search("[A-Za-z0-9]*\([A-Z]*\)", x):
#                 nodes.append(x)
#             else:
#                 print("wrong syntax")
#                 exit(0)
#         for y in nodes:
#             tet_tree.add_node(y)
#         for i, z in enumerate(edges):
#             # print(i)
#             # print(z)
#             tet_tree.add_edge(nodes[i], nodes[i+1], attr=z)
#         print(nodes)
#         print(edges)
#     return tet_tree
#         # if len(tet_spec) <= 1:
#         #     print("ya dumb bitch")
#         #     exit(0)
#         # for x in range(len(tet_spec)):
#         #     if x % 2 == 0:


# graph = TETspec("Yes(A) | -(A)-> | Bitch(A)")
# nx.draw(graph, pos=nx.spring_layout(graph))
# plt.show()

# nltk_tokens = nltk.word_tokenize("([fsdfdfg])")
# print(nltk_tokens)
# def count_of_counts(graph=nx.DiGraph()):
#     for node in graph:
#         if graph.out_degree(node) == 0 and graph.in_degree(node) >= 1:
#             leaf_count = nx.set_node_attributes(graph, 'count', 1)
#             graph.node[node]['count']
#
#         elif graph.out_degree(node) >= 1 and graph.in_degree(node) >=1:










