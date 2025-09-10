#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.pytest -p python313Packages.loguru
from __future__ import annotations

import sys
from collections import deque
from collections.abc import Hashable
from pathlib import Path
from typing import TypeVar

sys.path.append(Path(__file__).parent)
from graph import Graph, Node

H = TypeVar("H", bound=Hashable)


def bfs(graph: Graph[H], start: H) -> list[Node[H]]:
    if start not in graph.nodes:
        return []

    graph.reset_nodes()

    s = graph.nodes[start]
    visited = {s}
    queue = deque([s])
    result: list[Node[H]] = []
    time = 0
    s.start_time = time

    while queue:
        node = queue.popleft()
        result.append(node)
        node.end_time = time
        for n in node.adj:
            if n not in visited:
                n.predecessor = node
                n.start_time = time
                visited.add(n)
                queue.append(n)
        time += 1
    return result


if __name__ == "__main__":
    graph = Graph()
    graph.add_edge(0, 1)
    graph.add_edge(0, 2)
    graph.add_edge(1, 2)
    graph.add_edge(2, 3)

    result = bfs(graph, 0)
    print("BFS traversal:", [node.key for node in result])

    for node in result:
        print(
            f"Node {node.key}: start={node.start_time}, end={node.end_time}, predecessor={node.predecessor.key if node.predecessor else None}"
        )
