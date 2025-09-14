#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.pytest -p python313Packages.loguru
from __future__ import annotations

import sys
from collections.abc import Callable, Collection, Sequence
from pathlib import Path

from loguru import logger

sys.path.append(Path(__file__).parent)
from graph import Graph, H, Node


def dfs(
    graph: Graph[H],
    start: H | None = None,
    preorder: bool = False,
    rank: Callable[[Node[H]], int] = lambda n: n.key,
) -> Collection[Sequence[Node[H]]]:
    graph.reset_nodes()

    time = 0
    remaining_nodes = dict.fromkeys(sorted(graph.nodes.values(), key=rank))
    if start is None:
        start = next(iter(remaining_nodes.keys()))
    else:
        if start not in graph.nodes:
            return []
    s = graph.nodes[start]
    cc = s
    stack = [s]
    result = []

    def add_to_result(node: Node[H]):
        result.append(node)
        remaining_nodes.pop(node.key)
        node.cc = cc

    while stack or remaining_nodes:
        logger.trace(f"{time=}")
        if not stack:
            node = graph.nodes[next(iter(remaining_nodes))]
            cc = node
            logger.trace(f"New component: {cc.key}")
        else:
            node = stack.pop()

        if node.start_time is not None:
            node.end_time = time
            if not preorder:
                add_to_result(node)
        else:
            node.start_time = time
            stack.append(node)
            for n in filter(
                lambda n: n.start_time is None
                and n.predecessor is None
                and n.key in remaining_nodes,
                node.adj if rank is None else sorted(node.adj, key=rank),
            ):
                n.predecessor = node
                stack.append(n)
                if preorder:
                    add_to_result(n)
        time += 1
    return result


if __name__ == "__main__":
    graph = Graph()
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
