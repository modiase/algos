from __future__ import annotations
from typing import Any


class Tree:
    value: Any
    left: Tree | None
    right: Tree | None

    def __init__(self, value, left: Tree | None = None, right: Tree | None = None):
        self.value = value
        self.left = left
        self.right = right


def is_valid_btree(tree: Tree):
    return (
        tree.left is None
        or (tree.value > tree.left.value and is_valid_btree(tree.left))
    ) and (
        tree.right is None
        or (tree.right.value > tree.value and is_valid_btree(tree.right))
    )


t1 = Tree(5, Tree(10), Tree(20))
t2 = Tree(10, Tree(5), Tree(20))
t3 = Tree(10, Tree(5, Tree(2)), Tree(20, None, Tree(30)))

assert not is_valid_btree(t1)
assert is_valid_btree(t2)
assert is_valid_btree(t3)
