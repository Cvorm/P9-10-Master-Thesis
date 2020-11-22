import math
import numpy as np
from anytree import Node, RenderTree
from collections import Counter
from networkx import *


class Multiset:
    def __init__(self):
        g = nx.DiGraph()
        self.graph = g

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

    # def __count_helper(self,node):
    #     curr_node = self.graph.nodes[node]
    #     for n in self.graph.neighbors(curr_node):
    #
    # def count_of_counts(self):
    #     root = [x for x, y in self.graph.nodes(data=True) if y.get('root')]
    #     self.graph.nodes(data=True)[root]['count'] = len(self.graph.edges(root))
    #     for n in self.graph.neighbors(root):
    #         self.__count_helper(n)

    def __count_tree(self, pred):
        predecessors = [list(self.graph.predecessors(node)) for node in pred]
        final = list(set(self.__flat_list(predecessors)))
        if len(pred) > 0:
            for x in pred:
                print(f'pred: {x}')
                for p in final:
                    print(f'final: {final}')
                    if self.graph.nodes(data=True)[p].get('free'):
                        combined = [(self.graph.nodes(data=True)[x]['count'], self.graph.nodes(data=True)[x]['mult'])]
                        self.graph.nodes(data=True)[p]['mult'] += combined
                    else:
                        self.graph.nodes(data=True)[p]['mult'] += self.graph.nodes(data=True)[x]['count']
            self.__count_tree(final)
        else:
            for x in pred:
                print('YOOOO')
    @staticmethod
    def __flat_list(lst):
        flat_list = []
        for sublist in lst:
            for item in sublist:
                flat_list.append(item)
        return flat_list
    def __count_t(self, curr_node, leafs):
        if curr_node in leafs:
            self.graph.nodes(data=True)[curr_node]['mult'] = self.graph.nodes(data=True)[curr_node]['count']
            #leaves are directly evaluated by the Boolean value of their atom)
        elif self.graph.nodes(data=True)[curr_node].get('free'):
            succesors = list(self.graph.successors(curr_node))
            ls = []
            ls2 = []
            #DER SKAL TILFØJES NOGET SÅ VI HAR ET COUNT FOR HVER TYPE AF NODE
            for s in succesors:
                self.__count_t(s,leafs)
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
    def count_tree(self):
        root = [x for x, y in self.graph.nodes(data=True) if y.get('root')]
        leaf_nodes = [node for node in self.graph.nodes if
                      (self.graph.in_degree(node) != 0 and self.graph.out_degree(node) == 0)]
        self.__count_t(root[0],leaf_nodes)
    # def count_tree(self):
    #     leaf_nodes = [node for node in self.graph.nodes if
    #                   (self.graph.in_degree(node) != 0 and self.graph.out_degree(node) == 0)]
    #     predecessors = [list(self.graph.predecessors(node)) for node in leaf_nodes]
    #     final = list(set(self.__flat_list(predecessors)))
    #     self.__count_tree(leaf_nodes)
    #     freevar = [x for x, y in self.graph.nodes(data=True) if y.get('free')]
    #     print(freevar)
    #     root = [x for x, y in self.graph.nodes(data=True) if y.get('root')]
    #     for z in freevar:
    #         ree = list(self.graph.nodes(data=True)[z].get('mult'))
    #         # if len(ree) <= 1: continue
    #         # else:
    #         #     print(ree)
    #             # ree = list(self.graph.nodes(data=True)[z]['mult'])
    #         for ind, (n, m) in enumerate(ree):
    #             for ind2, (nn, mm) in enumerate(ree):
    #                 if n == nn:
    #                     v = n + 1
    #                     ree[ind] = (nn, v)
    #                     ree[ind2] = (nn, v)
    #         self.graph.nodes(data=True)[z]['mult'] = ree
    #     # ree = list(self.graph.nodes(data=True)[root[0]]['mult'])
    #     # for ind, (n, m) in enumerate(ree):
    #     #     for ind2, (nn, mm) in enumerate(ree):
    #     #         if n == nn:
    #     #             v = n + 1
    #     #             ree[ind] = (nn, v)
    #     #             ree[ind2] = (nn, v)
    #     #self.graph.nodes(data=True)[root[0]]['mult'] = ree

    def __sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def __set_weight(self, n, w):

        if self.graph.has_node(n):
            self.graph.nodes(data=True)[n]['weight'] = w
        else:
            (e1, e2) = n
            if self.graph.has_edge(e1, e2):
                self.graph[e1][e2]['weight'] = w

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

    def logistic_eval(self, bias, weight):
        # set weight root, and iterate through each edge from root to internal node
        leaf_nodes = [node for node in self.graph.nodes if
                      (self.graph.in_degree(node) != 0 and self.graph.out_degree(node) == 0)]
        root = [x for x, y in self.graph.nodes(data=True) if y.get('root')]
        self.__logistic_eval(root[0], bias, weight, leaf_nodes)

    def __histogram(self, node, leafs):
        if not node in leafs:
            h = []
            for x in self.graph.neighbors(node):
                h.append(self.graph.nodes(data=True)[x]['value'])
            h_unique_set = set(h)
            h_unique_list = list(h_unique_set)
            h_unique_list.sort()
            if len(h_unique_list) <= 1:
                hist, bin_edges = np.histogram(h, bins='auto')
                histogram = [list(self.normalize_list(hist)), list(bin_edges)]
                self.graph.nodes(data=True)[node]['hist'] += histogram
            else:
                hist, bin_edges = np.histogram(h, bins=h_unique_list)
                histogram = [list(self.normalize_list(hist)), list(bin_edges)]
                self.graph.nodes(data=True)[node]['hist'] += histogram
            for x in self.graph.neighbors(node):
                self.__histogram(x, leafs)

    def histogram(self):
        leaf_nodes = [node for node in self.graph.nodes if
                      (self.graph.in_degree(node) != 0 and self.graph.out_degree(node) == 0)]
        root = [x for x, y in self.graph.nodes(data=True) if y.get('root')]
        self.__histogram(root[0], leaf_nodes)

    def normalize_list(self, list):
        norm = [float(i) / sum(list) for i in list]
        return norm

    def same_length_lists(self, list1, list2):
        while len(list1) != len(list2):
            list1.append(0)
        return list1

    def emd_1d_histogram_similarity(self, hist1, hist2):
        #hist1 and hist2 must have the same length
        hist_w_padding = []
        dist = 0.0
        if len(hist1) < len(hist2):
            hist_w_padding = self.same_length_lists(hist1, hist2)
            dist = self.__compute_manhatten_distance(hist_w_padding, hist2)
        elif len(hist1) > len(hist2):
            hist_w_padding = self.same_length_lists(hist2, hist1)
            dist = self.__compute_manhatten_distance(hist_w_padding, hist1)
        else:
            dist = self.__compute_manhatten_distance(hist1, hist2)

        return dist

    def __compute_manhatten_distance(self, hist1, hist2):
        print(hist1, hist2)
        sum_list = []
        for x, y in zip(hist1, hist2):
            sum_list.append(abs(x - y))
        distance = sum(sum_list)
        return distance

    def __get_random_pair(self):
        print(self)


    def __splitdata(self):
        print(self)

    def __mtbuild(self, d_max, b_max, d):
        print(self)

    def mtbuild(self):
        self.__mtbuild(1, 1, 1)

        print(self)

    def __mtsearch(self):
        print(self)

    def mtsearch(self):
        print(self)


mult = Multiset()
list1 = [0.2, 0.3, 1, 1, 1]
list2 = [1, 1, 1, 0.6]
yes = mult.emd_1d_histogram_similarity(list1, list2)
print(yes)
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
