#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.pytest -p python313Packages.loguru -p python313Packages.more-itertools
"""
1. DFS(G).
2. Sort the nodes by end time.
3. DFS(G^T) choosing neighbours in the order of the sorted nodes.
4. The connected components are the sets of nodes with the same cc from step 3.
"""

import sys
from collections.abc import Collection, Sequence
from pathlib import Path

import pytest
from more_itertools import split_when

sys.path.append(Path(__file__).parent)
from dfs2 import dfs
from graph import Graph, H, Node


def components(graph: Graph[H]) -> Collection[Sequence[Node[H]]]:
    if len(graph) == 0:
        return []
    result = sorted(dfs(graph), key=lambda node: node.end_time, reverse=True)
    return split_when(
        dfs(graph.transpose(), rank=lambda node: result.index(node)),
        lambda n1, n2: n1.cc != n2.cc,
    )


@pytest.mark.parametrize(
    "graph,expected_sccs",
    [
        (
            Graph(
                edges=[
                    ("a", "b"),
                    ("b", "c"),
                    ("b", "e"),
                    ("b", "f"),
                    ("c", "d"),
                    ("c", "g"),
                    ("d", "c"),
                    ("d", "h"),
                    ("e", "a"),
                    ("e", "f"),
                    ("f", "g"),
                    ("g", "f"),
                    ("g", "h"),
                    ("h", "h"),
                ]
            ),
            {"abe", "cd", "fg", "h"},
        ),
        (
            Graph(
                edges=[
                    ("a", "j"),
                    ("a", "e"),
                    ("a", "b"),
                    ("b", "g"),
                    ("b", "j"),
                    ("b", "c"),
                    ("e", "g"),
                    ("e", "h"),
                    ("e", "j"),
                    ("e", "f"),
                    ("c", "g"),
                    ("c", "j"),
                    ("c", "f"),
                    ("g", "h"),
                    ("f", "g"),
                    ("d", "g"),
                    ("d", "e"),
                    ("d", "h"),
                    ("h", "j"),
                    ("h", "i"),
                ]
            ),
            {"a", "b", "c", "d", "e", "f", "g", "h", "j", "i"},
        ),
        (
            Graph(
                edges=[
                    ("a", "b"),
                    ("b", "c"),
                    ("c", "a"),
                ]
            ),
            {"abc"},
        ),
        (
            Graph(
                edges=[
                    ("a", "b"),
                    ("b", "c"),
                    ("b", "a"),
                ]
            ),
            {"ab", "c"},
        ),
        (
            Graph(
                edges=[
                    ("a", "b"),
                    ("b", "c"),
                    ("c", "d"),
                    ("c", "a"),
                    ("d", "e"),
                    ("e", "f"),
                    ("f", "d"),
                ]
            ),
            {"abc", "def"},
        ),
    ],
)
def test_components(graph: Graph[str], expected_sccs: set[str]) -> None:
    actual_sccs = set(
        map(
            lambda nodes: "".join(sorted([node.key for node in nodes])),
            components(graph),
        )
    )
    assert actual_sccs == expected_sccs


if __name__ == "__main__":
    pytest.main([__file__] + sys.argv[1:])
