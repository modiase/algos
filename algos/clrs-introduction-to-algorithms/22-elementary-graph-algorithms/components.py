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
    result = sorted(dfs(graph), key=lambda node: node.end_time)

    g_t = graph.transpose()
    return split_when(
        dfs(g_t, rank=lambda node: result.index(node)), lambda n1, n2: n1.cc != n2.cc
    )


@pytest.fixture
def graph() -> Graph[str]:
    graph = Graph()
    graph.add_edge("a", "b")
    graph.add_edge("b", "c")
    graph.add_edge("b", "e")
    graph.add_edge("b", "f")
    graph.add_edge("c", "d")
    graph.add_edge("c", "g")
    graph.add_edge("d", "c")
    graph.add_edge("d", "h")
    graph.add_edge("e", "a")
    graph.add_edge("e", "f")
    graph.add_edge("f", "g")
    graph.add_edge("g", "f")
    graph.add_edge("g", "h")
    graph.add_edge("h", "h")

    return graph


def test_components(graph: Graph[str]) -> None:
    assert set(
        map(
            lambda nodes: "".join(sorted([node.key for node in nodes])),
            components(graph),
        )
    ) == {"abe", "cd", "fg", "h"}


if __name__ == "__main__":
    pytest.main([__file__] + sys.argv[1:])
