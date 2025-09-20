#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.loguru -p python313Packages.pytest
from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).parent))
from graph import Graph, H


def relax(distances: dict[H, float], u: H, v: H, weight: float) -> bool:
    if distances[u] + weight < distances[v]:
        distances[v] = distances[u] + weight
        return True
    return False


def bellman_ford(graph: Graph[H], start: H) -> dict[H, float]:
    distances: dict[H, float] = {node.key: float("inf") for node in graph}
    distances[start] = 0

    for _ in range(len(graph) - 1):
        relaxed = False
        for node, neighbour, weight in graph.edges:
            relaxed |= relax(distances, node.key, neighbour.key, weight)
        if not relaxed:
            break

    for node, neighbour, weight in graph.edges:
        if distances[node.key] + weight < distances[neighbour.key]:
            raise ValueError("Graph contains a negative weight cycle")

    return distances


@pytest.mark.parametrize(
    "graph, start, expected",
    [
        (
            Graph(
                edges=[
                    (0, 1, 1),
                    (1, 0, 1),
                    (0, 2, 4),
                    (2, 0, 4),
                    (1, 2, 2),
                    (2, 1, 2),
                    (1, 3, 7),
                    (3, 1, 7),
                    (2, 3, 3),
                    (3, 2, 3),
                ]
            ),
            0,
            {0: 0, 1: 1, 2: 3, 3: 6},
        ),
        (
            Graph(
                edges=[
                    (0, 1, 1),
                    (1, 0, 1),
                    (0, 2, 4),
                    (2, 0, 4),
                    (1, 2, 2),
                    (2, 1, 2),
                    (1, 3, 7),
                    (3, 1, 7),
                    (2, 3, 3),
                    (3, 2, 3),
                ]
            ),
            1,
            {0: 1, 1: 0, 2: 2, 3: 5},
        ),
        (
            Graph(
                edges=[
                    (0, 1, 1),
                    (1, 0, 1),
                    (0, 2, 4),
                    (2, 0, 4),
                    (1, 2, 2),
                    (2, 1, 2),
                    (1, 3, 7),
                    (3, 1, 7),
                    (2, 3, 3),
                    (3, 2, 3),
                ]
            ),
            2,
            {0: 3, 1: 2, 2: 0, 3: 3},
        ),
        (
            Graph(
                edges=[
                    (0, 1, 1),
                    (1, 0, 1),
                    (0, 2, 4),
                    (2, 0, 4),
                    (1, 2, 2),
                    (2, 1, 2),
                    (1, 3, 7),
                    (3, 1, 7),
                    (2, 3, 3),
                    (3, 2, 3),
                ]
            ),
            3,
            {0: 6, 1: 5, 2: 3, 3: 0},
        ),
    ],
)
def test_bellman_ford(graph: Graph[H], start: H, expected: Mapping[H, float]):
    assert bellman_ford(graph, start) == expected


if __name__ == "__main__":
    pytest.main([__file__] + sys.argv[1:])
