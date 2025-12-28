#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.pytest
"""
Tests for splay tree implementation.
"""

from __future__ import annotations

import pytest

from splay import (
    Node,
    calculate_average_depth,
    count_nodes,
    get_depth,
    inorder,
    insert,
    is_root,
    left_rotate,
    right_rotate,
    search,
    splay,
)


def test_empty_tree() -> None:
    assert inorder(None) == []
    assert count_nodes(None) == 0
    assert get_depth(None, 5) is None


def test_single_insert() -> None:
    root = insert(None, 5)
    assert root.value == 5
    assert root.left is None
    assert root.right is None
    assert inorder(root) == [5]


def test_multiple_inserts_sorted() -> None:
    root = None
    for val in [1, 2, 3, 4, 5]:
        root = insert(root, val)
    assert inorder(root) == [1, 2, 3, 4, 5]
    assert root.value == 5


def test_multiple_inserts_reverse() -> None:
    root = None
    for val in [5, 4, 3, 2, 1]:
        root = insert(root, val)
    assert inorder(root) == [1, 2, 3, 4, 5]
    assert root.value == 1


def test_insert_duplicates_ignored() -> None:
    root = None
    for val in [5, 3, 7, 3, 5]:
        root = insert(root, val)
    assert inorder(root) == [3, 5, 7]
    assert count_nodes(root) == 3


def test_left_rotate() -> None:
    node = Node(value=1, right=Node(value=3, left=Node(value=2), right=Node(value=4)))
    rotated = left_rotate(node)
    assert rotated.value == 3
    assert rotated.left is not None
    assert rotated.left.value == 1
    assert rotated.left.right is not None
    assert rotated.left.right.value == 2


def test_right_rotate() -> None:
    node = Node(value=4, left=Node(value=2, left=Node(value=1), right=Node(value=3)))
    rotated = right_rotate(node)
    assert rotated.value == 2
    assert rotated.right is not None
    assert rotated.right.value == 4
    assert rotated.right.left is not None
    assert rotated.right.left.value == 3


def test_search_found() -> None:
    root = None
    for val in [5, 3, 7, 2, 4, 6, 8]:
        root = insert(root, val)

    new_root, found = search(root, 3)
    assert found is True
    assert new_root is not None
    assert new_root.value == 3


def test_search_not_found() -> None:
    root = None
    for val in [5, 3, 7]:
        root = insert(root, val)

    new_root, found = search(root, 10)
    assert found is False


def test_search_splays_to_root() -> None:
    root = None
    for val in [5, 3, 7, 2, 4, 6, 8]:
        root = insert(root, val)

    new_root, found = search(root, 2)
    assert found is True
    assert is_root(new_root, 2)


def test_splay_existing_value() -> None:
    root = None
    for val in [5, 3, 7, 2, 4]:
        root = insert(root, val)

    splayed = splay(root, 3)
    assert splayed is not None
    assert splayed.value == 3


def test_splay_nonexistent_brings_closest() -> None:
    root = None
    for val in [5, 3, 7]:
        root = insert(root, val)

    splayed = splay(root, 4)
    assert splayed is not None
    assert splayed.value in [3, 5]


def test_get_depth() -> None:
    root = Node(
        value=5,
        left=Node(value=3, left=Node(value=2), right=Node(value=4)),
        right=Node(value=7),
    )

    assert get_depth(root, 5) == 0
    assert get_depth(root, 3) == 1
    assert get_depth(root, 7) == 1
    assert get_depth(root, 2) == 2
    assert get_depth(root, 4) == 2
    assert get_depth(root, 10) is None


def test_is_root() -> None:
    root = Node(value=5, left=Node(value=3), right=Node(value=7))

    assert is_root(root, 5) is True
    assert is_root(root, 3) is False
    assert is_root(root, 7) is False
    assert is_root(None, 5) is False


def test_calculate_average_depth() -> None:
    root = Node(
        value=5,
        left=Node(value=3, left=Node(value=2), right=Node(value=4)),
        right=Node(value=7),
    )

    avg = calculate_average_depth(root)
    assert avg == (0 + 1 + 1 + 2 + 2) / 5


def test_calculate_average_depth_empty() -> None:
    assert calculate_average_depth(None) == 0.0


def test_count_nodes() -> None:
    root = Node(
        value=5,
        left=Node(value=3, left=Node(value=2)),
        right=Node(value=7, right=Node(value=8)),
    )
    assert count_nodes(root) == 5


@pytest.mark.parametrize(
    "values, expected_sorted",
    [
        ([3, 1, 2], [1, 2, 3]),
        ([5, 3, 7, 1, 9], [1, 3, 5, 7, 9]),
        ([10], [10]),
        ([2, 1, 3, 1, 2, 3], [1, 2, 3]),
    ],
)
def test_insert_various_patterns(values: list[int], expected_sorted: list[int]) -> None:
    root = None
    for val in values:
        root = insert(root, val)
    assert inorder(root) == expected_sorted


def test_sequential_searches_demonstrate_splaying() -> None:
    root = None
    for val in range(1, 8):
        root = insert(root, val)

    root, _ = search(root, 3)
    assert is_root(root, 3)

    root, _ = search(root, 5)
    assert is_root(root, 5)

    root, _ = search(root, 1)
    assert is_root(root, 1)


def test_immutability_insert() -> None:
    root1 = Node(value=5, left=Node(value=3), right=Node(value=7))
    _ = insert(root1, 4)

    assert root1.value == 5
    assert root1.left is not None
    assert root1.left.value == 3
    assert root1.left.right is None


def test_immutability_search() -> None:
    root1 = Node(value=5, left=Node(value=3), right=Node(value=7))
    root2, _ = search(root1, 3)

    assert root1.value == 5
    assert root2 is not None
    assert root2.value == 3
