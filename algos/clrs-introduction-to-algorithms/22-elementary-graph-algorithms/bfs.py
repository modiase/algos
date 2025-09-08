#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.pytest -p python313Packages.loguru
from __future__ import annotations

from collections import deque
from collections.abc import Hashable, Mapping

from loguru import logger


class Node:
    def __init__(self, key: Hashable):
        self.key = key
        self.adjacency_list = set()
        self._predecessor = None

    @property
    def predecessor(self) -> Node | None:
        return self._predecessor

    @predecessor.setter
    def predecessor(self, predecessor: Node) -> None:
        logger.trace(
            f"Setting predecessor of {self.key} to {predecessor.key if predecessor else None}"
        )
        self._predecessor = predecessor

    def __eq__(self, other: object):
        if isinstance(other, Node):
            return self.key == other.key
        elif isinstance(other, self.key.__class__):
            return self.key == other
        return NotImplemented

    def __hash__(self):
        return hash(self.key)

    def add_neighbor(self, neighbor: Node):
        logger.trace(f"Adding neighbor {neighbor.key} to {self.key}")
        self.adjacency_list.add(neighbor)

    def __str__(self):
        return f"Node({self.key})"

    def __repr__(self):
        return f"Node({self.key})"


class Graph:
    def __init__(self):
        self.nodes: Mapping[Hashable, Node] = {}

    def add_edge(self, k_u: Hashable, k_v: Hashable):
        logger.trace(f"Adding edge {k_u} -> {k_v}")
        u = self.nodes.setdefault(k_u, Node(k_u))
        v = self.nodes.setdefault(k_v, Node(k_v))
        u.add_neighbor(v)
        v.add_neighbor(u)

    def reset_nodes(self):
        for node in self.nodes.values():
            node.predecessor = None


def bfs(graph: Graph, start: Hashable):
    if start not in graph.nodes:
        return []

    graph.reset_nodes()

    s = graph.nodes[start]
    visited = {s}
    queue = deque([s])
    result = []

    while queue:
        node = queue.popleft()
        result.append(node)
        for n in node.adjacency_list:
            if n not in visited:
                n.predecessor = node
                visited.add(n)
                queue.append(n)
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
            f"Node {node.key}: predecessor={node.predecessor.key if node.predecessor else None}"
        )
