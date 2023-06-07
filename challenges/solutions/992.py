"""
Given the root to a binary search tree, find the second largest node in the tree.
"""


def dfs(t):
    if t.left is None:
        return [t.left]
    return dfs(t.left) + [t.val]


def find_second_smallest(t):
    return dfs(t)[1]


class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


t0 = Node(5, Node(3, Node(2), None), None)
t1 = Node(5, Node(3, Node(2, Node(1), Node(3)), None), None)

assert(find_second_smallest(t0)) == 3
assert(find_second_smallest(t1)) == 2
