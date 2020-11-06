from anytree import Node, RenderTree
from collections import Counter
from networkx import *


class Multiset:
    def __init__(self):
        g = nx.DiGraph()
        self.graph = g

    def add_root(self,node, c):
        self.graph.add_node(node, count=c, mult=[],root=True)

    def add_node_w_count(self,node,c):
        if not self.graph.has_node(node):
            self.graph.add_node(node,count=c, mult=0)

    def add_node(self, node):
        self.graph.add_node(node, count=1, mult=0)

    def add_nodes(self, node):
        self.graph.add_nodes_from(node, count=0, mult=0)

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

    def __bar(self, pred):
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
            self.__bar(final)


    @staticmethod
    def __flat_list(lst):
        flat_list = []
        for sublist in lst:
            for item in sublist:
                flat_list.append(item)
        return flat_list

    def foo(self):
        leaf_nodes = [node for node in self.graph.nodes if (self.graph.in_degree(node) != 0 and self.graph.out_degree(node) == 0)]
        predecessors = [list(self.graph.predecessors(node)) for node in leaf_nodes]
        final = list(set(self.__flat_list(predecessors)))
        self.__bar(final)
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


# class Multiset:
#     tree = []
#     root = Node("")
#     leaves = Counter()
#     def get_multiset(self):
#         return self.multi
#
#     def set_multiset(self, x):
#         self.multi = x
#
#     def __init__(self, root):
#         self.root = Node(root)
#
#     def add_leaf(self, genre):
#         self.leaves[genre] += 1
#
#     def add_internal_w_leaves(self, node):
#         count = Counter()
#         count[node] += 1
#         subtree = []
#         subtree.append(count)
#         subtree.append([self.leaves])
#         self.leaves = Counter()
#         return subtree
#
#     def add_internal(self, node, list):
#         count = Counter()
#         count[node] += 1
#         subtree = []
#         subtree.append(count)
#         res = []
#         for node in list:
#             res.append(node)
#         subtree.append([res])
#         return subtree
    # def add_leaves(self, list):
    #     list_leaves = []
    #     for count in list:
    #         list_leaves
    # def add_internal_with_leaves(self, node):
    #     if(len(self.leaves) < 1):
    #         print("No leaves to add")
    #     count = Counter()
    #     count[node] += 1

# multi = Multiset("user1")
# multi.add_leaf("action")
# multi.add_leaf("bromance")
# yes = multi.add_internal_w_leaves("movie")
# yesss = multi.add_internal("user", yes)
# print(yesss)