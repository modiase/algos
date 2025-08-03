"""
Given a binary tree, find a minimum path sum from root to a leaf.

For example, the minimum path in this tree is [10, 5, 1, -1], which has sum 15.

  10
 /  \
5    5
 \     \
   2    1
       /
     -1

Review
================
Completed in 20m

Notes
================
Initially used bfs assuming I could exit early because I didn't
read question properly. Initially implemented path length
instead of path sum. Then realised that I had to consider all
possible paths after all. Luckily this was simple enough to fix
by removing early return. Overall, should have completed this much
more quickly.

"""

from __future__ import annotations

from collections import deque
from heapq import heappop, heappush
from typing import Deque, Optional, Tuple, List


class Node:
    def __init__(
        self, value: int, left: Optional[Node] = None, right: Optional[Node] = None
    ):
        self._value = value
        self._left = left
        self._right = right

    @property
    def is_leaf(self) -> bool:
        return self._left == None and self._right == None


def min_path_sum_to_leaf(root: Node) -> int:
    leaf_path_sums: List[int] = []
    queue: Deque[Tuple[int, Node]] = deque([(0, root)])
    while len(queue) != 0:
        (sum, node) = queue.popleft()

        sum = sum + node._value
        if node.is_leaf:
            heappush(leaf_path_sums, sum)

        if node._left is not None:
            queue.append((sum, node._left))
        if node._right is not None:
            queue.append((sum, node._right))

    return heappop(leaf_path_sums)


t0 = Node(10)
assert min_path_sum_to_leaf(t0) == 10

t1 = Node(10, Node(5, Node(2)), Node(5, right=Node(1, Node(-1))))
assert min_path_sum_to_leaf(t1) == 15

t2 = Node(10, Node(5))
assert min_path_sum_to_leaf(t2) == 15

t3 = Node(10, Node(5, Node(2, Node(1, Node(0)))))
assert min_path_sum_to_leaf(t3) == 18

t4 = Node(10, Node(-5, Node(2, Node(1, Node(0)))), Node(2))
assert min_path_sum_to_leaf(t4) == 8

t5 = Node(10, Node(-5, Node(2, Node(1, Node(0)))), Node(-3))
assert min_path_sum_to_leaf(t5) == 7
