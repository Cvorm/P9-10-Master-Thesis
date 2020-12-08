import math
import numpy as np
from anytree import Node, RenderTree
from collections import Counter
from networkx import *


# This structure contains a TET and a metric tree for each user
class Multiset:
    def __init__(self):
        g = nx.DiGraph()
        mt = nx.DiGraph()
        self.graph = g
        self.mt = mt

    def add_root(self, node, c):
        self.graph.add_node(node, count=c, mult=[], root=True, weight=0, value=0, hist=[], free=True, type='user')

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
            #leaves are directly evaluated by the Boolean value of their atom)
        elif self.graph.nodes(data=True)[curr_node].get('free'):    # if our current node is a free variable, then we calculate the count of counts
            succesors = list(self.graph.successors(curr_node))
            ls = []
            ls2 = []
            #DER SKAL TILFØJES NOGET SÅ VI HAR ET COUNT FOR HVER TYPE AF NODE
            for s in succesors:
                self.__count_tree(s, leafs)
                ls.append(self.graph.nodes(data=True)[s]['mult'])
                ls2.append(self.graph.nodes(data=True)[s]['type'])
                # combined = [(self.graph.nodes(data=True)[s]['mult'], self.graph.nodes(data=True)[curr_node]['count'])]
                # self.graph.nodes(data=True)[curr_node]['mult'] += combined
            types = set(ls2)
            res = dict.fromkeys(types, 0)
            if not any(isinstance(i, list) for i in ls):
                for s in succesors:
                    for t in types:
                        if self.graph.nodes(data=True)[s]['type'] == t:
                            res[t] += 1
                for x in res.items():
                    self.graph.nodes(data=True)[curr_node]['mult'] += [x]
            else:
                res = {}
                for x in ls:
                    #res[f'{x}'] = 0
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
            # print('NOT')
            self.__set_weight(node, bias)
            for n in self.graph.neighbors(node):
                self.__logistic_eval(n, bias, weight, leafs)
            var = [self.graph.nodes(data=True)[n]['value'] for n in self.graph.neighbors(node)]  # .count(1)
            # nei = list(self.graph.neighbors(node))
            # rn = 0
            # for n in nei:
            #     print(n)
            #     m = self.graph.nodes(data=True)[n]['mult']
            #     if type(m) == int:
            #         rn += 1
            #         print('SINGLE INT')
            #         print(m)
            #     elif any(isinstance(i, tuple) for i in m):
            #         rn += 1
            #         print('LIST OF LIST')
            #         print(m)
            #     else:
            #         rn += 1
            #         print('ELSE')
            #         print(m)


            c = 0
            for x in var:
                c += x
            # v_b = bias + (len(self.graph.out_edges(node)) * weight) * var
            v_b = bias + weight * c
            v = self.__sigmoid(v_b)
            self.graph.nodes(data=True)[node]['value'] = v

        if node in leafs:
            # print('LEAF')
            self.__set_weight(node, 1)
            self.graph.nodes(data=True)[node]['value'] = weight

    # call function for logistic evalution
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




# mult = Multiset()
# list1 = [0.2, 0.3, 1, 1, 1]
# list2 = [1, 1, 1, 0.6]
# yes = mult.emd_1d_histogram_similarity(list1, list2)
# print(yes)
#example graph
# ms = Multiset()
# ms.add_root('u1')
# ms.add_node('m1')
# ms.add_node('m2')
# ms.add_node('action')
# ms.add_node('comedy')
# ms.add_node('bromance')
# ms.add_edge(('u1','m1'))
# ms.add_edge(('u1','m2'))
# ms.add_edge(('m2','action'))
# ms.add_edge(('m2','comedy'))
# ms.add_edge(('m1','action'))
# ms.add_edge(('m1','comedy'))
# ms.add_edge(('m1','bromance'))
#
# ms.foo()
