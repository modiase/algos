#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.loguru
from __future__ import annotations

import sys
from collections.abc import Hashable, Sequence
from pathlib import Path
from typing import TypeVar

from loguru import logger

sys.path.append(Path(__file__).parent)
from graph import Graph, Node

H = TypeVar("H", bound=Hashable)


def dfs_topological_sort(graph: Graph[H]) -> Sequence[Node[H]]:
    graph.reset_nodes()
    time = 0
    result: list[Node[H]] = []
    nodes = {n.key: n for n in graph}
    while nodes:
        start = next(iter(nodes.values()))
        start.predecessor = start
        stack = [start]
        while stack:
            node = stack.pop()
            logger.trace(f"Popped node from stack: {node.key}")
            if node.start_time is None:
                node.start_time = time
                stack.append(node)
                for neighbor in node.adj:
                    if neighbor.predecessor is None:
                        neighbor.predecessor = node
                        stack.append(neighbor)
            else:
                node.end_time = time
                result.append(node)
                nodes.pop(node.key)

            time += 1
    return sorted(result, key=lambda node: node.end_time)


if __name__ == "__main__":
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
    result = dfs_topological_sort(graph)
    for node in result:
        print(
            f"Node {node.key}: start={node.start_time}, end={node.end_time}, predecessor={node.predecessor.key if node.predecessor else None}"
        )
