#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.pytest -p python313Packages.loguru
from __future__ import annotations

import sys
from pathlib import Path
from typing import assert_never

from loguru import logger

sys.path.append(Path(__file__).parent)
from graph import Graph, H, Node2


def dfs(graph: Graph[H], start: H):
    """Ilustrative algorithm using three colors to mark the state of the nodes."""
    if start not in graph.nodes:
        return []

    graph.reset_nodes()

    time = 0
    stack = [graph.nodes[start]]
    result = []

    while stack:
        logger.trace(f"Stack: {stack} at time {time}")
        node = stack.pop()
        logger.trace(f"Popped node from stack: {node.key}")
        match node.state:
            case Node2.Color.WHITE:
                node.state = Node2.Color.GRAY
                node.start_time = time
                time += 1

                logger.trace(f"Pushing node {node.key} back to stack")
                stack.append(node)

                for neighbor in node.adj:
                    if neighbor.state == Node2.Color.WHITE:
                        logger.trace(
                            f"Pushing neighbor {neighbor.key} of {node.key} to stack"
                        )
                        neighbor.predecessor = node
                        stack.append(neighbor)
            case Node2.Color.GRAY:
                node.state = Node2.Color.BLACK
                node.end_time = time
                time += 1
                result.append(node)
            case Node2.Color.BLACK:
                pass
            case never:
                assert_never(never)

    return result


if __name__ == "__main__":
    graph = Graph(node_class=Node2)
    graph.add_edge(0, 1)
    graph.add_edge(0, 2)
    graph.add_edge(1, 2)
    graph.add_edge(2, 3)

    result = dfs(graph, 0)
    print("DFS traversal:", [node.key for node in result])

    for node in result:
        print(
            f"Node {node.key}: start={node.start_time}, end={node.end_time}, predecessor={node.predecessor.key if node.predecessor else None}"
        )
