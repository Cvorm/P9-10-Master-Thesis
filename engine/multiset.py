import math
import numpy as np
from networkx import *


# This structure contains a TET and a metric tree for each user
class Multiset:
    def __init__(self):
        g = nx.DiGraph()
        mt = nx.DiGraph()
        ht = nx.DiGraph()
        self.graph = g
        self.mt = mt
        self.ht = ht

    def add_root(self, node, c):
        self.graph.add_node(node, count=c, mult=[], root=True, weight=0, value=0, hist=[], free=True, type='user')

    def add_node_w_cross(self, node, t, cross):
        tmp_count = cross.count(True)
        self.graph.add_node(node, count=tmp_count, mult=[], weight=0, value=0, hist=[], type=t, cross=cross)

    def add_node_w_count(self, node, c, t):
        if not self.graph.has_node(node):
            self.graph.add_node(node, count=c, mult=0, weight=0, value=0, hist=[], type=t)

    def add_node(self, node):
        self.graph.add_node(node, count=0, mult=0, weight=0, value=0, hist=[])

    def add_nodes(self, node):
        self.graph.add_nodes_from(node, count=0, mult=0, weight=0, value=0, hist=[])

    def add_node_w_freevar(self, node, c, t):
        self.graph.add_node(node, count=c, mult=[], weight=0, value=0, hist=[],free=True, type=t)

    def add_edge(self, edge):
        (v1, v2) = edge
        self.graph.add_edge(v1, v2, weight=0)

    def add_edges(self, edge):
        self.graph.add_edges_from(edge, weight=0)

    def get_graph(self):
        return self.graph

    def set_graph(self, g):
        self.graph = g

    def get_histogram(self,node):
        return self.graph.nodes(data=True)[node]['hist']

    # def get_logistic_values(self,node):
    #     for n in node:
    #
    #     return self.graph.nodes(data=True)[node]

    @staticmethod
    def __flat_list(lst):
        flat_list = []
        for sublist in lst:
            for item in sublist:
                flat_list.append(item)
        return flat_list

    # helper function for counting the tree
    def __count_tree(self, curr_node, leafs):
        if curr_node in leafs:  # if our current node is a leaf, then the count is simply the count
            self.graph.nodes(data=True)[curr_node]['mult'] = self.graph.nodes(data=True)[curr_node]['count']
            # if self.graph.nodes(data=True)[curr_node]['type'] == 'genre':
            #     self.graph.nodes(data=True)[curr_node]['mult'] += self.graph.nodes(data=True)[curr_node]['cross']
            # if self.graph.nodes(data=True)[curr_node]['type'] != 'genre':
            #     self.graph.nodes(data=True)[curr_node]['mult'] = self.graph.nodes(data=True)[curr_node]['count']
        elif self.graph.nodes(data=True)[curr_node].get('free'):    # if our current node is a free variable, then we calculate the count of counts
            succesors = list(self.graph.successors(curr_node))
            ls = []
            ls2 = []
            ls3 = []
            for s in succesors:
                self.__count_tree(s, leafs)
                ls.append(self.graph.nodes(data=True)[s]['mult'])
                if self.graph.nodes(data=True)[s]['type'] == 'genre':
                    ls2.append(s)
                else:
                    ls2.append(self.graph.nodes(data=True)[s]['type'])
                # ls3.append([self.graph.nodes(data=True)[s]['type'], self.graph.nodes(data=True)[s]['mult']])
                # self.graph.nodes(data=True)[curr_node]['mult'] += combined
            types = set(ls2)
            res = dict.fromkeys(types, 0)
            if not any(isinstance(i, list) for i in ls):    # if we are dealing we a list of list of count
                for s in succesors:
                    for t in types:
                        if self.graph.nodes(data=True)[s]['type'] == t:
                            res[t] += 1
                        elif self.graph.nodes(data=True)[s]['type'] == 'genre':
                            if t == s:
                                res[t] += 1
                for x in res.items():
                    self.graph.nodes(data=True)[curr_node]['mult'] += [x]
            else:   # if we are dealing with a single count
                res = {}
                for x in ls:
                    res[tuple(x)] = 0
                    for j in ls:
                        if x == j:
                            #print(f'X: {x} J: {j}')
                            res[tuple(x)] += 1
                for x in res.items():
                    self.graph.nodes(data=True)[curr_node]['mult'] += [x]

    # call function for counting the tree
    def count_tree(self):
        root = [x for x, y in self.graph.nodes(data=True) if y.get('root')]
        leaf_nodes = [node for node in self.graph.nodes if
                      (self.graph.in_degree(node) != 0 and self.graph.out_degree(node) == 0)]
        self.__count_tree(root[0], leaf_nodes)

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
        edges = list(self.graph.out_edges(node))
        [self.__set_weight(e, weight) for e in edges]
        if not node in leafs:
            self.__set_weight(node, bias)
            for n in self.graph.neighbors(node):
                self.__logistic_eval(n, bias, weight, leafs)
            value_list = [self.graph.nodes(data=True)[n]['value'] for n in self.graph.neighbors(node)]  # .count(1)
            temp_count = 0
            for value in value_list:
                temp_count += value
            # v_b = bias + (len(self.graph.out_edges(node)) * weight) * var
            v_b = bias + (weight * temp_count)
            v = self.__sigmoid(v_b)
            self.graph.nodes(data=True)[node]['value'] = v

        if node in leafs:
            self.__set_weight(node, 1)
            self.graph.nodes(data=True)[node]['value'] = self.graph.nodes(data=True)[node]['count']

    # call function for logistic evaluation
    def logistic_eval(self, bias, weight):
        # set weight root, and iterate through each edge from root to internal node
        leaf_nodes = [node for node in self.graph.nodes if
                      (self.graph.in_degree(node) != 0 and self.graph.out_degree(node) == 0)]
        root = [x for x, y in self.graph.nodes(data=True) if y.get('root')]
        self.__logistic_eval(root[0], bias, weight, leaf_nodes)

    # function for calculating histograms
    def __histogram(self, node, leafs, num_bins):
        if not node in leafs:
            h = []
            for x in self.graph.neighbors(node):
                h.append(self.graph.nodes(data=True)[x]['value'])
            h_unique_set = set(h)
            h_unique_list = list(h_unique_set)
            h_unique_list.sort()
            if len(h_unique_list) <= 1:
                hist, bin_edges = np.histogram(h, bins='auto')
                histogram = [list(hist), list(bin_edges)]
                self.graph.nodes(data=True)[node]['hist'] += histogram
            else:
                hist, bin_edges = np.histogram(h, bins=num_bins)
                histogram = [list(hist), list(bin_edges)]
                self.graph.nodes(data=True)[node]['hist'] += histogram
            for x in self.graph.neighbors(node):
                self.__histogram(x, leafs, num_bins)

    # call function for histogram
    def histogram(self, num_bins):
        leaf_nodes = [node for node in self.graph.nodes if
                      (self.graph.in_degree(node) != 0 and self.graph.out_degree(node) == 0)]
        root = [x for x, y in self.graph.nodes(data=True) if y.get('root')]
        self.__histogram(root[0], leaf_nodes, num_bins)


    def __histogram2(self, node, leafs, specif):
        self.ht.add_node(node, hist=[])
        h = []
        if node in ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'IMAX', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']:
            tmp = self.graph.in_degree(node)
            try:
                d = int(tmp)
                for n in range(d):
                    h.append(1)
            except:
                pass
        else:
            nodes = [x for x,y in self.graph.nodes(data=True) if y.get('type') == node]
            for n in nodes:
                h.append(self.graph.nodes(data=True)[n]['value'])
        hist, bin_edges = np.histogram(h, bins=[0.0,0.2,0.4,0.6,0.8,1.0]) # [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
        histogram = [list(hist), list(bin_edges)]
        self.ht.nodes(data=True)[node]['hist'] += histogram
        for x in specif.neighbors(node):
            self.__histogram2(x, leafs,specif)
            self.ht.add_edge(node,x)

    def histogram2(self,specif):
        leaf_nodes = [node for node in self.graph.nodes if
                      (self.graph.in_degree(node) != 0 and self.graph.out_degree(node) == 0)]
        root = [x for x, y in self.graph.nodes(data=True) if y.get('root')]
        r = [x for x in specif.nodes if x == 'user']
        self.__histogram2(r[0], leaf_nodes,specif)

    @staticmethod
    def __normalize_list(list):
        norm = [float(i) / sum(list) for i in list]
        return norm

