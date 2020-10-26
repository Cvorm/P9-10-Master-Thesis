from collections import Counter
from anytree import Node, RenderTree
import networkx as nx
import matplotlib.pyplot as plt
import re
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

def TETspec(tet_spec):
    tet_tree = nx.Graph()
    tokens = []
    nodes = []
    edges = []
    if len(tet_spec.split(" | ")) <= 2:
        print("dude")
        exit(0)
    else:
        tokens = tet_spec.split(" | ")
        # for x in range(len(tokens)):
        for x in tokens:
            if re.search("(-(\([A-Z]*\))->)", x):
                edges.append(x)
            elif re.search("[A-Za-z0-9]*\([A-Z]*\)", x):
                nodes.append(x)
            else:
                print("wrong syntax")
                exit(0)
        for y in nodes:
            tet_tree.add_node(y)
        for i, z in enumerate(edges):
            # print(i)
            # print(z)
            tet_tree.add_edge(nodes[i], nodes[i+1], attr=z)
        print(nodes)
        print(edges)
    return tet_tree
        # if len(tet_spec) <= 1:
        #     print("ya dumb bitch")
        #     exit(0)
        # for x in range(len(tet_spec)):
        #     if x % 2 == 0:


graph = TETspec("Yes(A) | -(A)-> | Bitch(A)")
nx.draw(graph, pos=nx.spring_layout(graph))
plt.show()


def count_of_counts(graph=nx.DiGraph()):
    for node in graph:
        if graph.out_degree(node) == 0 and graph.in_degree(node) >= 1:
            leaf_count = nx.set_node_attributes(graph, 'count', 1)
            graph.node[node]['count']

        elif graph.out_degree(node) >= 1 and graph.in_degree(node) >=1:










