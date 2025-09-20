#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.loguru -p python313Packages.pytest
from __future__ import annotations

import sys
from collections.abc import Mapping, MutableMapping
from heapq import heappop, heappush
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).parent))
from graph import Graph, H


def relax(distances: MutableMapping[H, float], u: H, v: H, weight: float) -> bool:
    if distances[u] + weight < distances[v]:
        distances[v] = distances[u] + weight
        return True
    return False


def dijkstra(graph: Graph[H], start: H) -> Mapping[H, float]:
    """
    Not optimised. This implementation is O((V+E) log V).
    There are optimised implementations that are O(E + V log V) using exotic heaps (e.g. Fibonacci heap).
    """
    graph.add_node(start)

    distances: MutableMapping[H, float] = {node.key: float("inf") for node in graph}
    distances[start] = 0

    pq: list[tuple[float, H]] = [(0, start)]
    processed: set[H] = set()

    while pq:
        _, u = heappop(pq)

        if u in processed:
            continue

        processed.add(u)

        for neighbor, weight in graph[u].adj:
            if neighbor.key not in processed and relax(
                distances, u, neighbor.key, weight
            ):
                heappush(pq, (distances[neighbor.key], neighbor.key))

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
        (
            Graph(edges=[(0, 1, 5), (2, 3, 3)]),
            0,
            {0: 0, 1: 5, 2: float("inf"), 3: float("inf")},
        ),
        (
            Graph(edges=[(0, 1, 5), (1, 2, 3), (2, 3, 2)]),
            1,
            {0: float("inf"), 1: 0, 2: 3, 3: 5},
        ),
    ],
)
def test_dijkstra(graph: Graph[H], start: H, expected: Mapping[H, float]):
    result = dijkstra(graph, start)
    assert result == expected


if __name__ == "__main__":
    pytest.main([__file__] + sys.argv[1:])
