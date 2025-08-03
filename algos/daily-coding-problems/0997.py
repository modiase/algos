"""
Given the root to a binary tree, implement serialize(root), which serializes the tree into a string, and deserialize(s), which deserializes the string back into the tree.

For example, given the following Node class

class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
The following test should pass:

node = Node('root', Node('left', Node('left.left')), Node('right'))
assert deserialize(serialize(node)).left.left.val == 'left.left'
"""

import re


def serialize(t):
    if t is None:
        return "None"
    return f"Node('{t.val}', {serialize(t.left)}, {serialize(t.right)})"


class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


node = Node("root", Node("left", Node("left.left")), Node("right"))


def deserialize(s):
    if s == "None":
        return None
    pattern = r"Node\(\'(?P<value>.+?)\', (?P<left>(?:None|Node\(.*\))), (?P<right>(?:None|Node\(.*\)))\)"
    match = re.search(pattern, s)
    if match is None:
        raise Exception(s)
    val, left, right = match.groups()
    return Node(val, deserialize(left), deserialize(right))


print(serialize(deserialize(serialize(node))))
assert deserialize(serialize(node)).left.left.val == "left.left"
