#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.pytest -p python313Packages.loguru
from __future__ import annotations

from collections.abc import Hashable, Mapping

from loguru import logger


class Node:
    def __init__(self, key: Hashable):
        self.key = key
        self.adjacency_list = set()
        self._is_processed = False
        self._start_time = None
        self._end_time = None
        self._predecessor = None

    @property
    def start_time(self) -> int | None:
        return self._start_time

    @start_time.setter
    def start_time(self, time: int) -> None:
        logger.trace(f"Setting start time of {self.key} to {time}")
        self._start_time = time

    @property
    def end_time(self) -> int | None:
        return self._end_time

    @end_time.setter
    def end_time(self, time: int) -> None:
        logger.trace(f"Setting end time of {self.key} to {time}")
        self._end_time = time

    @property
    def is_processed(self) -> bool:
        return self._is_processed

    @is_processed.setter
    def is_processed(self, state: bool):
        logger.trace(f"Setting is_processed of {self.key} to {state}")
        self._is_processed = state

    @property
    def predecessor(self) -> Node | None:
        return self._predecessor

    @predecessor.setter
    def predecessor(self, predecessor: Node) -> None:
        logger.trace(f"Setting predecessor of {self.key} to {predecessor.key}")
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


def dfs(graph: Graph, start: Hashable):
    if start not in graph.nodes:
        return []

    for node in graph.nodes.values():
        node.is_processed = False

    time = 0
    s = graph.nodes[start]
    s.start_time = time
    stack = [s]
    result = []

    while stack:
        logger.trace(f"{time=}")
        node = stack.pop()
        if node.is_processed:
            node.end_time = time
        else:
            node.is_processed = True
            node.start_time = time
            result.append(node)
            stack.append(node)
            for n in filter(
                lambda n: not n.is_processed and n.predecessor is None,
                node.adjacency_list,
            ):
                n.predecessor = node
                stack.append(n)
        time += 1
    return result


if __name__ == "__main__":
    graph = Graph()
    graph.add_edge(0, 1)
    graph.add_edge(0, 2)
    graph.add_edge(1, 2)
    graph.add_edge(2, 3)

    result = dfs(graph, 0)
    print("DFS with processing bit traversal:", [node.key for node in result])

    for node in result:
        print(
            f"Node {node.key}: start={node.start_time}, end={node.end_time}, predecessor={node.predecessor.key if node.predecessor else None}"
        )
