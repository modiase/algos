#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.pytest
from __future__ import annotations

from dataclasses import dataclass
import sys

import pytest


@dataclass(slots=True)
class TreeNode:
    val: int
    left: TreeNode | None = None
    right: TreeNode | None = None


def lowestCommonAncestor(root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
    def dfs(current: TreeNode | None, val: int, path: str) -> str | None:
        if current is None:
            return None
        if current.val == val:
            return path
        if (pth := dfs(current.left, val, path + "l")) is not None:
            return pth
        return dfs(current.right, val, path + "r")

    p_path = dfs(root, p.val, "")
    q_path = dfs(root, q.val, "")
    if p_path is None:
        raise ValueError("p not found")
    if q_path is None:
        raise ValueError("q not found")

    idx = 0
    while idx < min(len(p_path), len(q_path)):
        if p_path[idx] != q_path[idx]:
            break
        idx += 1

    current = root
    for c in p_path[:idx]:
        if c == "l":
            current = current.left
        else:
            current = current.right

    return current


def _sample_tree() -> tuple[
    TreeNode, TreeNode, TreeNode, TreeNode, TreeNode, TreeNode, TreeNode
]:
    r"""
    Construct the following tree:
              3
            /   \
           5     1
          / \   / \
         6   2 0   8
            / \
           7   4
    """

    node7 = TreeNode(7, None, None)
    node4 = TreeNode(4, None, None)
    node6 = TreeNode(6, None, None)
    node2 = TreeNode(2, node7, node4)
    node5 = TreeNode(5, node6, node2)
    node0 = TreeNode(0, None, None)
    node8 = TreeNode(8, None, None)
    node1 = TreeNode(1, node0, node8)
    root = TreeNode(3, node5, node1)
    return root, node5, node1, node6, node2, node7, node4


@pytest.mark.parametrize(
    ("p_idx", "q_idx", "expected_idx"),
    [
        (1, 2, 0),  # 5 and 1 -> 3
        (1, 3, 1),  # 5 and 6 -> 5
        (4, 6, 4),  # 2 and 4 -> 2
        (3, 5, 1),  # 6 and 7 -> 5
        (1, 1, 1),  # same node -> self
    ],
)
def test_lowest_common_ancestor(p_idx: int, q_idx: int, expected_idx: int) -> None:
    nodes = _sample_tree()
    root = nodes[0]
    p = nodes[p_idx]
    q = nodes[q_idx]
    expected = nodes[expected_idx]

    assert lowestCommonAncestor(root, p, q) is expected


if __name__ == "__main__":
    pytest.main([__file__, *sys.argv[1:]])
