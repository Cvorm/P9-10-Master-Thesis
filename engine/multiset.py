import math
import numpy as np
from anytree import Node, RenderTree
from collections import Counter
from networkx import *


class Multiset:
    def __init__(self):
        g = nx.DiGraph()
        self.graph = g

    def add_root(self,node, c):
        self.graph.add_node(node, count=c, mult=[],root=True,weight=0,value=0,hist=[])

    def add_node_w_count(self,node,c):
        if not self.graph.has_node(node):
            self.graph.add_node(node,count=c, mult=0,weight=0,value=0,hist=[])

    def add_node(self, node):
        self.graph.add_node(node, count=0, mult=0,weight=0,value=0,hist=[])

    def add_nodes(self, node):
        self.graph.add_nodes_from(node, count=0, mult=0,weight=0,value=0,hist=[])

    def add_edge(self, edge):
        (v1,v2) = edge
        self.graph.add_edge(v1,v2, weight=0)

    def add_edges(self, edge):
        self.graph.add_edges_from(edge, weight=0)

    def get_graph(self):
        return self.graph

    def set_graph(self, g):
        self.graph = g

    def __count_helper(self,node):
        curr_node = self.graph.nodes[node]
        for n in self.graph.neighbors(curr_node):
            print(n)

    def count_of_counts(self):
        root = [x for x, y in self.graph.nodes(data=True) if y.get('root')]
        self.graph.nodes(data=True)[root]['count'] = len(self.graph.edges(root))
        for n in self.graph.neighbors(root):
            self.__count_helper(n)
            print(n)

    def __count_tree(self, pred):
        predecessors = [list(self.graph.predecessors(node)) for node in pred]
        final = list(set(self.__flat_list(predecessors)))
        if len(pred) > 0:
            for x in pred:
                for p in final:
                    if self.graph.nodes(data=True)[p]['root']:
                        combined = [(self.graph.nodes(data=True)[x]['count'],self.graph.nodes(data=True)[x]['mult'])]
                        self.graph.nodes(data=True)[p]['mult'] += combined
                    else:
                        self.graph.nodes(data=True)[p]['mult'] += self.graph.nodes(data=True)[x]['mult']
            self.__count_tree(final)


    @staticmethod
    def __flat_list(lst):
        flat_list = []
        for sublist in lst:
            for item in sublist:
                flat_list.append(item)
        return flat_list

    def count_tree(self):
        leaf_nodes = [node for node in self.graph.nodes if (self.graph.in_degree(node) != 0 and self.graph.out_degree(node) == 0)]
        predecessors = [list(self.graph.predecessors(node)) for node in leaf_nodes]
        final = list(set(self.__flat_list(predecessors)))
        self.__count_tree(final)
        root = [x for x, y in self.graph.nodes(data=True) if y.get('root')]
        print(root)
        ree = list(self.graph.nodes(data=True)[root[0]]['mult'])
        for ind,(n,m) in enumerate(ree):
            for ind2, (nn,mm) in enumerate(ree):
                if n == nn:
                    v = n + 1
                    ree[ind] = (nn, v)
                    ree[ind2] = (nn,v)
        self.graph.nodes(data=True)[root[0]]['mult'] = ree
        print(ree)
        print(self.graph.nodes(data=True))

    def __sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def __set_weight(self,n,w):

        if self.graph.has_node(n):
            self.graph.nodes(data=True)[n]['weight'] = w
        else:
            (e1,e2) = n
            if self.graph.has_edge(e1,e2):
                self.graph[e1][e2]['weight'] = w

    def __logistic_eval(self, node, bias, weight, leafs):
        edges = list(self.graph.out_edges(node))
        [self.__set_weight(e,weight) for e in edges]
        if not node in leafs:
            #print('NOT')
            self.__set_weight(node, bias)
            for n in self.graph.neighbors(node):
                self.__logistic_eval(n, bias, weight, leafs)
            var = [self.graph.nodes(data=True)[n]['value'] for n in self.graph.neighbors(node)] #.count(1)
            c = 0
            for x in var:
                c += x
            #v_b = bias + (len(self.graph.out_edges(node)) * weight) * var
            v_b = bias + weight * c
            v = self.__sigmoid(v_b)
            self.graph.nodes(data=True)[node]['value'] = v

        if node in leafs:
            #print('LEAF')
            self.__set_weight(node,1)
            self.graph.nodes(data=True)[node]['value'] = weight

    def logistic_eval(self,bias,weight):
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
            # myset = set(h)
            # mysett = list(myset)
            # mysett.sort()
            hist, bin_edges = np.histogram(h, bins='auto')
            histogram = []
            histogram.append(list(hist))
            histogram.append(list(bin_edges))
            print(histogram)
            self.graph.nodes(data=True)[node]['hist'] += histogram
            for x in self.graph.neighbors(node):
                self.__histogram(x, leafs)

    def histogram(self):
        leaf_nodes = [node for node in self.graph.nodes if
                      (self.graph.in_degree(node) != 0 and self.graph.out_degree(node) == 0)]
        root = [x for x, y in self.graph.nodes(data=True) if y.get('root')]
        self.__histogram(root[0], leaf_nodes)
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
