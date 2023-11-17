"""
https://leetcode.com/explore/interview/card/top-interview-questions-medium/108/trees-and-graphs/786/
Given the root of a binary tree, return the inorder traversal of its nodes' values.

Notes
=====
## Summary
T: 5
C: Y
PD: 1

## Comments

Fundamental algorithm

tags: trees
"""


class TreeNode:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def inorder_traversal(node: TreeNode):
    def _dfs(node):
        if node is None:
            return []
        return _dfs(node.left) + [node.val] + _dfs(node.right)
    return _dfs(node)


assert inorder_traversal(TreeNode(0)) == [0]
assert inorder_traversal(TreeNode(2, TreeNode(1), TreeNode(3))) == [1, 2, 3]
assert inorder_traversal(TreeNode(2, TreeNode(
    1, TreeNode(0, TreeNode(-1))), TreeNode(3))) == [-1, 0, 1, 2, 3]
