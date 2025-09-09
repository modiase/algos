#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.loguru
from __future__ import annotations

from collections.abc import Collection, Hashable, Iterator, Mapping, Sequence
from typing import Generic, TypeVar

from loguru import logger

H = TypeVar("H", bound=Hashable)


class Node(Generic[H]):
    class AdjacencyList(Generic[H]):
        def __init__(self):
            self.nodes = set()

        def add(self, node: Node[H]):
            self.nodes.add(node)

        def __iter__(self):
            return iter(sorted(self.nodes, key=lambda node: node.key, reverse=True))

    def __init__(self, key: H):
        self.key = key
        self.adjacency_list = self.AdjacencyList()
        self._start_time = None
        self._end_time = None
        self._predecessor = None

    @property
    def predecessor(self) -> Node[H] | None:
        return self._predecessor

    @predecessor.setter
    def predecessor(self, predecessor: Node[H]):
        logger.trace(
            f"Setting predecessor of {self.key} to {predecessor.key if predecessor else None}"
        )
        self._predecessor = predecessor

    @property
    def start_time(self) -> int | None:
        return self._start_time

    @start_time.setter
    def start_time(self, time: int):
        logger.trace(f"Setting start time of {self.key} to {time}")
        self._start_time = time

    @property
    def end_time(self) -> int | None:
        return self._end_time

    @end_time.setter
    def end_time(self, time: int):
        logger.trace(f"Setting end time of {self.key} to {time}")
        self._end_time = time

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Node):
            return self.key == other.key
        elif isinstance(other, self.key.__class__):
            return self.key == other
        return NotImplemented

    def __hash__(self):
        return hash(self.key)

    def add_neighbor(self, neighbor: Node[H]):
        logger.trace(f"Adding neighbor {neighbor.key} to {self.key}")
        self.adjacency_list.add(neighbor)

    def __str__(self):
        return f"Node({self.key})"

    def __repr__(self):
        return f"Node({self.key})"


class Graph(Generic[H]):
    def __init__(self, edges: Collection[tuple[H, H]] | None = None):
        self.nodes: Mapping[H, Node[H]] = {}
        for k_u, k_v in edges or []:
            self.add_edge(k_u, k_v)

    def add_edge(self, k_u: H, k_v: H):
        logger.trace(f"Adding edge {k_u} -> {k_v}")
        u = self.nodes.setdefault(k_u, Node(k_u))
        v = self.nodes.setdefault(k_v, Node(k_v))
        u.add_neighbor(v)

    def reset_nodes(self):
        for node in self.nodes.values():
            node.start_time = None
            node.end_time = None
            node.predecessor = None

    def __iter__(self) -> Iterator[Node[H]]:
        return iter(sorted(self.nodes.values(), key=lambda node: node.key))


def dfs_topological_sort(graph: Graph[H]) -> Sequence[Node[H]]:
    graph.reset_nodes()
    time = 0
    start = next(iter(graph), None)
    result = []
    if start is None:
        return result
    nodes = {n.key: n for n in graph}
    while nodes:
        stack = [next(iter(nodes.values()))]
        while stack:
            node = stack.pop()
            logger.trace(f"Popped node from stack: {node.key}")
            if node.start_time is None:
                node.start_time = time
                stack.append(node)
                for neighbor in node.adjacency_list:
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
