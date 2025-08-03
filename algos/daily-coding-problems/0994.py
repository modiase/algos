"""
Print the nodes in a binary tree level-wise. For example, the following should print 1, 2, 3, 4, 5.

  1
 / \
2   3
   / \
  4   5
  """


class Tree:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


t0 = Tree(1, Tree(2), Tree(3, Tree(4), Tree(5)))


def traverse_tree(t):
    def head(l):
        return [l[0]] if l else []

    left = traverse_tree(t.left) if t.left is not None else []
    right = traverse_tree(t.right) if t.right is not None else []
    if t.right is not None:
        right = traverse_tree(t.right)
    return [t.val, *head(left), *head(right), *left[1:], *right[1:]]


assert traverse_tree(t0) == [1, 2, 3, 4, 5]
