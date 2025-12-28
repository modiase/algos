"""
Splay tree implementation using functional/immutable style.

A splay tree is a self-adjusting binary search tree where recently accessed
elements are moved to the root through a series of tree rotations called splaying.

Key properties:
- Amortized O(log n) time for search, insert
- Recently accessed items stay near the root (good for caching)
- No balance information stored (unlike AVL trees)

Splaying operations (when moving target node to root):

> - right rotation at node
< - left rotation at node

1. Zig (single rotation) - when target's parent is the root:

   Splay 3:
       5>             3
      / \            / \
     3   7    =>    1   5
    /                     \
   1                      7

   Single right rotation at 5 brings 3 to root.

2. Zig-Zag (opposite directions) - when target zigzags (left-right or right-left):

   Splay 6:
       8              8>             6
      / \            / \            / \
     3<  9    =>    6   9    =>    3   8
      \            /                \   \
       6          3                  4   9
      /            \
     4              4

   Step 1: Left rotate at 3 (parent)
   Step 2: Right rotate at 8 (grandparent)

3. Zig-Zig (same direction) - when target, parent, grandparent form a line:

   Splay 1:
       7>             5>              3
      /              / \              \
     5              3   7              5
    /              /            =>    / \  => (zig-zag case)
   3              1                  1   7
    \              \                  \
     1              2                  2
      \
       2

   Step 1: Right rotate at 7 (grandparent first)
   Step 2: Right rotate at 5 (old parent)

   Key: Zig-zig rotates grandparent first, pulling up the entire left spine.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Node:
    value: int
    left: Node | None = None
    right: Node | None = None


def left_rotate(node: Node) -> Node:
    if node.right is None:
        return node
    return Node(
        value=node.right.value,
        left=Node(value=node.value, left=node.left, right=node.right.left),
        right=node.right.right,
    )


def right_rotate(node: Node) -> Node:
    if node.left is None:
        return node
    return Node(
        value=node.left.value,
        left=node.left.left,
        right=Node(value=node.value, left=node.left.right, right=node.right),
    )


def splay(root: Node | None, value: int) -> Node | None:
    """
    Splay the tree to bring value closer to root.

    Uses simplified splaying approach. Returns new root.

    Time: Amortized O(log n)
    """
    if root is None:
        return None

    if root.value == value:
        return root

    if value < root.value:
        if root.left is None:
            return root

        if value < root.left.value:
            if root.left.left:
                return right_rotate(
                    right_rotate(
                        Node(
                            value=root.value,
                            left=Node(
                                value=root.left.value,
                                left=splay(root.left.left, value),
                                right=root.left.right,
                            ),
                            right=root.right,
                        )
                    )
                )
            else:
                return right_rotate(root)

        elif value > root.left.value:
            if root.left.right:
                new_root_temp = Node(
                    value=root.value,
                    left=Node(
                        value=root.left.value,
                        left=root.left.left,
                        right=splay(root.left.right, value),
                    ),
                    right=root.right,
                )
                return right_rotate(
                    Node(
                        value=new_root_temp.value,
                        left=left_rotate(new_root_temp.left)
                        if new_root_temp.left
                        else None,
                        right=new_root_temp.right,
                    )
                )
            else:
                return right_rotate(root)

        else:
            return right_rotate(root)

    else:
        if root.right is None:
            return root

        if value > root.right.value:
            if root.right.right:
                return left_rotate(
                    left_rotate(
                        Node(
                            value=root.value,
                            left=root.left,
                            right=Node(
                                value=root.right.value,
                                left=root.right.left,
                                right=splay(root.right.right, value),
                            ),
                        )
                    )
                )
            else:
                return left_rotate(root)

        elif value < root.right.value:
            if root.right.left:
                new_root_temp = Node(
                    value=root.value,
                    left=root.left,
                    right=Node(
                        value=root.right.value,
                        left=splay(root.right.left, value),
                        right=root.right.right,
                    ),
                )
                return left_rotate(
                    Node(
                        value=new_root_temp.value,
                        left=new_root_temp.left,
                        right=right_rotate(new_root_temp.right)
                        if new_root_temp.right
                        else None,
                    )
                )
            else:
                return left_rotate(root)

        else:
            return left_rotate(root)


def insert(root: Node | None, value: int) -> Node:
    """
    Insert value into tree and splay it to root.

    Time: Amortized O(log n)
    """
    if root is None:
        return Node(value=value)

    splayed_root = splay(root, value)
    if splayed_root is None:
        return Node(value=value)

    if splayed_root.value == value:
        return splayed_root

    if value < splayed_root.value:
        return Node(
            value=value,
            left=splayed_root.left,
            right=Node(value=splayed_root.value, right=splayed_root.right),
        )
    else:
        return Node(
            value=value,
            left=Node(value=splayed_root.value, left=splayed_root.left),
            right=splayed_root.right,
        )


def search(root: Node | None, value: int) -> tuple[Node | None, bool]:
    """
    Search for value and splay it to root if found.

    Returns (new_root, found) where found is True if value exists.

    Time: Amortized O(log n)
    """
    if root is None:
        return None, False

    splayed = splay(root, value)
    return splayed, splayed is not None and splayed.value == value


def get_depth(root: Node | None, value: int) -> int | None:
    """
    Get depth of value in tree (root has depth 0).

    Returns None if value not found.

    Time: O(n) worst case
    """

    def helper(node: Node | None, target: int, depth: int) -> int | None:
        if node is None:
            return None
        if node.value == target:
            return depth
        if target < node.value:
            return helper(node.left, target, depth + 1)
        else:
            return helper(node.right, target, depth + 1)

    return helper(root, value, 0)


def is_root(root: Node | None, value: int) -> bool:
    """
    Check if value is at the root.

    Time: O(1)
    """
    return root is not None and root.value == value


def calculate_average_depth(root: Node | None) -> float:
    """
    Calculate average depth of all nodes in tree.

    Time: O(n)
    """

    def helper(node: Node | None, depth: int) -> tuple[int, int]:
        if node is None:
            return 0, 0

        left_sum, left_count = helper(node.left, depth + 1)
        right_sum, right_count = helper(node.right, depth + 1)

        total_sum = depth + left_sum + right_sum
        total_count = 1 + left_count + right_count

        return total_sum, total_count

    total_sum, total_count = helper(root, 0)
    return total_sum / total_count if total_count > 0 else 0.0


def count_nodes(root: Node | None) -> int:
    """
    Count total number of nodes in tree.

    Time: O(n)
    """
    if root is None:
        return 0
    return 1 + count_nodes(root.left) + count_nodes(root.right)


def inorder(root: Node | None) -> list[int]:
    """
    Return inorder traversal of tree.

    Time: O(n)
    """
    if root is None:
        return []
    return inorder(root.left) + [root.value] + inorder(root.right)
