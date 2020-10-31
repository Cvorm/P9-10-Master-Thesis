from anytree import Node, RenderTree
from collections import Counter

class Multiset:
    tree = []
    root = Node("")
    leaves = Counter()
    def get_multiset(self):
        return self.multi

    def set_multiset(self, x):
        self.multi = x

    def __init__(self, root):
        self.root = Node(root)

    def add_leaf(self, genre):
        self.leaves[genre] += 1

    def add_internal_w_leaves(self, node):
        count = Counter()
        count[node] += 1
        subtree = []
        subtree.append(count)
        subtree.append([self.leaves])
        self.leaves = Counter()
        return subtree

    def add_internal(self, node, list):
        count = Counter()
        count[node] += 1
        subtree = []
        subtree.append(count)
        res = []
        for node in list:
            res.append(node)
        subtree.append([res])
        return subtree
    # def add_leaves(self, list):
    #     list_leaves = []
    #     for count in list:
    #         list_leaves
    # def add_internal_with_leaves(self, node):
    #     if(len(self.leaves) < 1):
    #         print("No leaves to add")
    #     count = Counter()
    #     count[node] += 1

multi = Multiset("user1")
multi.add_leaf("action")
multi.add_leaf("bromance")
yes = multi.add_internal_w_leaves("movie")
yesss = multi.add_internal("user", yes)
print(yesss)