"""
Given a binary tree, return all paths from the root to leaves.

For example, given the tree:

   1
  / \
 2   3
    / \
   4   5
Return [[1, 2], [1, 3, 4], [1, 3, 5]].

Solved: 4m 51s
"""


class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def all_paths(t0):
    paths = []

    def _all_paths(acc, t):
        if t.left is None and t.right is None:
            paths.append([*acc, t.val])
            return
        _all_paths([*acc, t.val], t.left)
        _all_paths([*acc, t.val], t.right)

    _all_paths([], t0)

    return paths


t1 = Node(1, Node(2), Node(3, Node(4), Node(5)))
assert list(sorted(all_paths(t1))) == list(sorted([[1, 2], [1, 3, 4], [1, 3, 5]]))
