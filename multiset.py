from anytree import Node, RenderTree
from collections import Counter
from networkx import *


class Multiset:
    def __init__(self):
        g = nx.DiGraph()
        self.graph = g

    def add_root(self,node):
        self.graph.add_node(node, count=0, mult=0,root=True)

    def add_node_w_count(self,node,c):
        self.graph.add_node(node,count=c)

    def add_node(self, node):
        self.graph.add_node(node, count=0, mult=0)

    def add_nodes(self, node):
        self.graph.add_nodes_from(node, count=0, mult=0)

    def add_edge(self, edge):
        self.graph.add_edge(edge, weight=0)

    def add_edges(self, edge):
        self.graph.add_nodes_from(edge, weight=0)

    def get_graph(self):
        return self.graph

    def set_graph(self, g):
        self.graph = g

    def __count_helper(self,node):
        curr_node = self.graph.nodes[node]
        for n in self.graph.neighbors(curr_node):
            print(n)
        return self

    def count_of_counts(self):
        root = [x for x, y in self.graph.nodes(data=True) if y.get('root')]
        self.graph.nodes(data=True)[root]['count'] = len(self.graph.edges(root))
        for n in self.graph.neighbors(root):
            self.__count_helper(n)
            print(n)


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