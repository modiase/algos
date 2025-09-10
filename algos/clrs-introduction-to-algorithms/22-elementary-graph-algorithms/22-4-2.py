#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.pytest -p python313Packages.loguru
from __future__ import annotations

import sys
from collections.abc import Hashable
from pathlib import Path
from typing import TypeVar

import pytest
from loguru import logger

sys.path.append(Path(__file__).parent)
from graph import Graph

H = TypeVar("H", bound=Hashable)


def count_paths(graph: Graph[H], start: H, end: H) -> int:
    graph.reset_nodes()
    time = 0
    result = 0
    if (start := graph.nodes.get(start, None)) is None or (
        end := graph.nodes.get(end, None)
    ) is None:
        return result
    stack = [start]
    while stack:
        node = stack.pop()
        logger.trace(f"Popped node from stack: {node.key}")
        if node.start_time is None:
            node.start_time = time
            stack.append(node)
            for neighbor in node.adj:
                if neighbor.key == end:
                    logger.trace(f"Found end node {end} from {node.key}")
                    result += 1
                if neighbor.predecessor is None:
                    neighbor.predecessor = node
                    stack.append(neighbor)
        else:
            node.end_time = time

        time += 1
    return result


@pytest.fixture
def test_dag() -> Graph[str]:
    graph = Graph(
        [
            ("m", "q"),
            ("m", "r"),
            ("m", "x"),
            ("n", "q"),
            ("n", "o"),
            ("n", "u"),
            ("o", "r"),
            ("o", "s"),
            ("o", "v"),
            ("p", "s"),
            ("p", "z"),
            ("q", "t"),
            ("r", "u"),
            ("r", "y"),
            ("s", "r"),
            ("u", "t"),
            ("v", "w"),
            ("w", "z"),
            ("y", "v"),
        ]
    )
    return graph


@pytest.mark.parametrize(
    "start, end, expected",
    [
        ("m", "z", 1),
        ("m", "t", 2),
        ("m", "v", 1),
        ("m", "p", 0),
    ],
)
def test_count_paths(test_dag: Graph[str], start: str, end: str, expected: int):
    result = count_paths(test_dag, start, end)
    assert result == expected


if __name__ == "__main__":
    pytest.main([__file__] + sys.argv[1:])
