"""
Print the nodes in a binary tree level-wise. For example, the following should print
1, 2, 3, 4, 5.

  1
 / \
2   3
   / \
  4   5


Notes
========
Completed in 2m
"""

from __future__ import annotations

from typing import Optional
from collections import deque


class Node:
    def __init__(
        self, value: int, left: Optional[Node] = None, right: Optional[Node] = None
    ):
        self._value = value
        self._left = left
        self._right = right


def bfs(root):
    queue = deque([root])
    result = []
    while len(queue) != 0:
        node = queue.popleft()
        if node._left is not None:
            queue.append(node._left)
        if node._right is not None:
            queue.append(node._right)
        result.append(node._value)
    return result


t0 = Node(1, Node(2), Node(3, Node(4), Node(5)))
assert bfs(t0) == [1, 2, 3, 4, 5]
