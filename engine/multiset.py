import math
import numpy as np
from networkx import *


# This structure contains a TET and a metric tree for each user
class Multiset:
    def __init__(self):
        g = nx.DiGraph()
        ht = nx.DiGraph()
        self.graph = g
        self.ht = ht

    def add_root(self, node, c):
        self.graph.add_node(node, count=c, mult=[], root=True, value=0.0, hist=[], free=True, type='user', user=node)

    def add_root_movie(self, node, c):
        self.graph.add_node(node, count=c, mult=[], root=True, value=0, hist=[], free=True, type='movie', user=node)

    def add_node_w_count(self, node, c, t):
        if not self.graph.has_node(node):
            self.graph.add_node(node, count=c, mult=0, value=0.0, type=t)

    def add_node_w_count_w_val(self, node, c, v, t):
        if not self.graph.has_node(node):
            self.graph.add_node(node, count=c, mult=v, value=0.0, type=t)

    def add_node(self, node):
        self.graph.add_node(node, count=0.0, mult=0, value=0.0, hist=[])

    def add_nodes(self, node):
        self.graph.add_nodes_from(node, count=0, mult=0, value=0)

    def add_edge(self, edge):
        (v1, v2) = edge
        self.graph.add_edge(v1, v2, weight=0)

    def add_edges(self, edge):
        self.graph.add_edges_from(edge, weight=0)

    def get_graph(self):
        return self.graph

    def set_graph(self, g):
        self.graph = g

    def get_histogram(self):
        return self.ht.nodes(data=True)

    @staticmethod
    def __flat_list(lst):
        flat_list = []
        for sublist in lst:
            for item in sublist:
                flat_list.append(item)
        return flat_list


    def __sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    # helper function for setting the weight of edges
    def __set_weight(self, n, w):
        if self.graph.has_node(n):
            self.graph.nodes(data=True)[n]['weight'] = w
        else:
            (e1, e2) = n
            if self.graph.has_edge(e1, e2):
                self.graph[e1][e2]['weight'] = w

    # function for performing logistic evaluation
    def __logistic_eval(self, node, bias, weight, leafs):
        if not node in leafs:
            for n in self.graph.neighbors(node):
                self.__logistic_eval(n, bias, weight, leafs)
            value_list = [self.graph.nodes(data=True)[n]['value'] for n in self.graph.neighbors(node)]  # .count(1)
            temp_count = 0.0
            for value in value_list:
                temp_count += value
            # v_b = bias + (len(self.graph.out_edges(node)) * weight) * var
            #print(f'NODE: {node}')
            v_b = bias + (weight * temp_count)
            v = self.__sigmoid(v_b)
            # if self.graph.nodes(data=True)[node].get('root'):
            #     print(temp_count)
            #     print(f'v_b: {v_b}, res= {v}')

            #print(f'v_b: {v_b}, res= {v}')
            self.graph.nodes(data=True)[node]['value'] = v
        if node in leafs:
            self.graph.nodes(data=True)[node]['value'] = self.graph.nodes(data=True)[node]['count']

    # call function for logistic evaluation
    def logistic_eval(self, bias, weight):
        # set weight root, and iterate through each edge from root to internal node
        leaf_nodes = [node for node in self.graph.nodes if
                      (self.graph.in_degree(node) != 0 and self.graph.out_degree(node) == 0)]
        root = [x for x, y in self.graph.nodes(data=True) if y.get('root')]
        self.__logistic_eval(root[0], bias, weight, leaf_nodes)

    # function for calculating histograms
    def __histogram(self, node, leafs, specif):
        self.ht.add_node(node, hist=[])
        h = []
        nodes = [x for x, y in self.graph.nodes(data=True) if y.get('type') == specif.nodes(data=True)[node]['type']] # specif.nodes(data=True)[node]['type']
        # print(f'Nodes: {nodes}')
        for n in nodes:
            h.append(self.graph.nodes(data=True)[n]['value'])
        # print(f'hist: {h}')
        hist, bin_edges = np.histogram(h, bins=[0.0,0.2,0.4,0.6,0.8,1.0]) # [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0] [0.0,0.2,0.4,0.6,0.8,1.0]
        histogram = [list(hist), list(bin_edges)]
        self.ht.nodes(data=True)[node]['hist'] += histogram
        for x in specif.neighbors(node):
            self.__histogram(x, leafs, specif)
            self.ht.add_edge(node, x)

    def histogram(self, specif, root):
        leaf_nodes = [node for node in self.graph.nodes if
                      (self.graph.in_degree(node) != 0 and self.graph.out_degree(node) == 0)]
        # root = [x for x, y in self.graph.nodes(data=True) if y.get('root')]
        r = [x for x in specif.nodes if x == root]
        self.__histogram(r[0], leaf_nodes, specif)

    @staticmethod
    def __normalize_list(list):
        norm = [float(i) / sum(list) for i in list]
        return norm

