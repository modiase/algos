#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.pytest -p python313Packages.loguru
from __future__ import annotations

from collections.abc import Hashable, Mapping
from enum import IntEnum
from typing import assert_never

from loguru import logger


class Node:
    def __init__(self, key: Hashable):
        self.key = key
        self.adjacency_list = set()
        self._state = Color.WHITE
        self._start_time = None
        self._end_time = None
        self._predecessor = None

    @property
    def predecessor(self) -> Node | None:
        return self._predecessor

    @predecessor.setter
    def predecessor(self, predecessor: Node):
        logger.trace(f"Setting predecessor of {self.key} to {predecessor.key}")
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

    @property
    def state(self) -> Color:
        return self._state

    @state.setter
    def state(self, state: Color):
        logger.trace(f"Setting state of {self.key} to {state}")
        self._state = state

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
            node.state = Color.WHITE
            node.start_time = None
            node.end_time = None
            node.predecessor = None


class Color(IntEnum):
    WHITE = 0  # not visited
    GRAY = 1  # being visited
    BLACK = 2  # visited


def dfs(graph: Graph, start: Hashable):
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
            case Color.WHITE:
                node.state = Color.GRAY
                node.start_time = time
                time += 1

                logger.trace(f"Pushing node {node.key} back to stack")
                stack.append(node)

                for neighbor in node.adjacency_list:
                    if neighbor.state == Color.WHITE:
                        logger.trace(
                            f"Pushing neighbor {neighbor.key} of {node.key} to stack"
                        )
                        neighbor.predecessor = node
                        stack.append(neighbor)
            case Color.GRAY:
                node.state = Color.BLACK
                node.end_time = time
                time += 1
                result.append(node)
            case Color.BLACK:
                pass
            case never:
                assert_never(never)

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
