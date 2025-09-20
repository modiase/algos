#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.loguru -p python313Packages.pytest
from __future__ import annotations

import sys
from collections import deque
from collections.abc import Mapping, MutableMapping, MutableSequence
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).parent))
from graph import Graph, H


def topological_sort(graph: Graph[H]) -> MutableSequence[H]:
    in_degree: MutableMapping[H, int] = {node.key: 0 for node in graph}

    for _, neighbor, _ in graph.edges:
        in_degree[neighbor.key] += 1

    queue = deque([node.key for node in graph if in_degree[node.key] == 0])
    top_order: MutableSequence[H] = []

    while queue:
        u = queue.popleft()
        top_order.append(u)

        for neighbor, _ in graph[u].adj:
            in_degree[neighbor.key] -= 1
            if in_degree[neighbor.key] == 0:
                queue.append(neighbor.key)

    return top_order


def relax(distances: MutableMapping[H, float], u: H, v: H, weight: float) -> bool:
    if distances[u] + weight < distances[v]:
        distances[v] = distances[u] + weight
        return True
    return False


def dag_shortest_path(graph: Graph[H], start: H) -> Mapping[H, float]:
    distances: MutableMapping[H, float] = {node.key: float("inf") for node in graph}
    distances[start] = 0

    for u in topological_sort(graph):
        if distances[u] != float("inf"):
            # By the time we reach a node, all directed paths to that node have been relaxed.
            # Therefore, if it is unreachable, it will remain unreachable and we can skip it.
            for neighbor, weight in graph[u].adj:
                relax(distances, u, neighbor.key, weight)

    return distances


@pytest.mark.parametrize(
    "graph, start, expected",
    [
        (
            Graph(edges=[(0, 1, 5), (1, 2, 3), (2, 3, 2)]),
            0,
            {0: 0, 1: 5, 2: 8, 3: 10},
        ),
        (
            Graph(edges=[(0, 1, 1), (0, 2, 4), (1, 3, 2), (2, 3, 1)]),
            0,
            {0: 0, 1: 1, 2: 4, 3: 3},
        ),
        (
            Graph(edges=[(0, 1, 5), (1, 2, -2), (2, 3, 3), (0, 2, 2)]),
            0,
            {0: 0, 1: 5, 2: 2, 3: 5},
        ),
        (
            Graph(edges=[]),
            0,
            {0: 0},
        ),
        (
            Graph(edges=[(0, 1, 7)]),
            0,
            {0: 0, 1: 7},
        ),
        (
            Graph(
                edges=[
                    (0, 1, 2),
                    (0, 2, 6),
                    (1, 2, 3),
                    (1, 3, 1),
                    (2, 3, 4),
                    (2, 4, 2),
                    (3, 4, 1),
                    (3, 5, 3),
                    (4, 5, 2),
                ]
            ),
            0,
            {0: 0, 1: 2, 2: 5, 3: 3, 4: 4, 5: 6},
        ),
    ],
)
def test_dag_shortest_path(graph: Graph[H], start: H, expected: Mapping[H, float]):
    result = dag_shortest_path(graph, start)
    assert result == expected


if __name__ == "__main__":
    pytest.main([__file__] + sys.argv[1:])
