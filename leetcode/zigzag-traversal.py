"""

Notes
=====
## Summary
T: 20
C: Y
PD: 2

## Comments

Got a bit confused about the implementation and need to make sure I am keeping
in mind how when the list is reversed it is only reversed for the result, because
of the way I'm traversing each level, if I reverse the list before traversing then
you end up putting the children in the wrong order (left child or rightmost, right
child of rightmost, left child of second right most) and not (left child of leftmost
...) which results in the correct traversal when reversed.

tags: trees, traversal, b-trees
"""
from collections import deque
from typing import List, Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def zigzag_traverse(root: Optional[TreeNode]) -> List[List[int]]:
    result = []
    q = deque([[]])
    if root is None:
        return []
    result.append([root.val])
    if root.left is not None:
        q[0].append(root.left)
    if root.right is not None:
        q[0].append(root.right)

    leftToRight = False
    while q:
        l = q.popleft()
        if l:
            lc = l
            if not leftToRight:
                lc = reversed(l)
            result.append([n.val for n in lc])
        for n in l:
            if not q:
                q.append([])
            if n.left is not None:
                q[0].append(n.left)
            if n.right is not None:
                q[0].append(n.right)
        leftToRight = not leftToRight

    return result


assert zigzag_traverse(
    TreeNode(1, TreeNode(2, TreeNode(4)), TreeNode(3, right=TreeNode(5)))) == [[1], [3, 2], [4, 5]]
