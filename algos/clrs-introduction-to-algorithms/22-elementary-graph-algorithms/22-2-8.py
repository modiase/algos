#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.pytest -p python313Packages.loguru
from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import pytest
from loguru import logger

sys.path.append(Path(__file__).parent)
from graph import Graph, H, Node


def graph_diameter(graph: Graph[H]) -> int:
    start = next(iter(graph), None)
    if start is None:
        return 0

    def bfs_helper(start: Node[H]):
        node, distance = start, 0
        node.predecessor = node
        queue = deque([(node, distance)])
        while queue:
            node, distance = queue.popleft()
            logger.trace(f"Visiting node {node.key} at distance {distance}")
            for neighbor in node.adj:
                if neighbor.predecessor is None:
                    neighbor.predecessor = node
                    queue.append((neighbor, distance + 1))
        logger.trace(f"Returning node {node.key} at distance {distance}")
        return node, distance

    graph.reset_nodes()
    furthest, _ = bfs_helper(start)
    graph.reset_nodes()
    _, distance = bfs_helper(furthest)
    return distance


@pytest.mark.parametrize(
    "graph, expected",
    [
        (Graph(), 0),
        (Graph(edges=[(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)]), 3),
        (
            Graph(
                edges=[(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2), (3, 4), (4, 3)]
            ),
            4,
        ),
        (
            Graph(
                edges=[
                    (0, 1),
                    (1, 0),
                    (1, 2),
                    (2, 1),
                    (2, 3),
                    (3, 2),
                    (3, 4),
                    (4, 3),
                    (4, 5),
                    (5, 4),
                ]
            ),
            5,
        ),
        (Graph(edges=[(0, 1), (1, 0), (1, 2), (2, 1), (2, 0), (0, 2)]), 1),
        (
            Graph(
                edges=[(0, 1), (1, 0), (1, 2), (2, 1), (2, 0), (0, 2), (2, 3), (3, 2)]
            ),
            2,
        ),
    ],
)
def test_graph_diameter(graph: Graph[int], expected: int):
    assert graph_diameter(graph) == expected


if __name__ == "__main__":
    pytest.main([__file__, *sys.argv[1:]])
