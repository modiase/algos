from __future__ import annotations

import random as rn
from collections.abc import Collection, Hashable, Iterator, Mapping
from enum import IntEnum
from typing import Callable, Generic, TypeVar

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
        self.adj: Collection[Node[H]] = self.AdjacencyList()
        self._start_time: int | None = None
        self._end_time: int | None = None
        self._predecessor: Node[H] | None = None
        self._cc: Node[H] | None = None

    @property
    def cc(self) -> Node[H] | None:
        return self._cc

    @cc.setter
    def cc(self, cc: Node[H] | None):
        """The component to which this node belongs."""
        logger.trace(f"Setting cc of {self.key} to {cc.key if cc else None}")
        self._cc = cc

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
        self.adj.add(neighbor)

    def __str__(self):
        return str(self.key)

    def __repr__(self):
        return f"Node({self.key})"

    def reset_node(self):
        self.start_time = None
        self.end_time = None
        self.predecessor = None


class Node2(Node[H]):
    class Color(IntEnum):
        WHITE = 0  # not visited
        GRAY = 1  # being visited
        BLACK = 2  # visited

    def __init__(self, key: H):
        super().__init__(key)
        self._state = self.Color.WHITE

    @property
    def state(self) -> Color:
        return self._state

    @state.setter
    def state(self, state: Color):
        logger.trace(f"Setting state of {self.key} to {state}")
        self._state = state


class Graph(Generic[H]):
    def __init__(
        self,
        edges: Collection[tuple[H, H]] | None = None,
        node_class: type[Node[H]] = Node,
    ):
        self.nodes: Mapping[H, Node[H]] = {}
        self.node_class = node_class
        for k_u, k_v in edges or []:
            self.add_edge(k_u, k_v)

    def add_node(self, key: H):
        self.nodes[key] = self.node_class(key)

    def __getitem__(self, key: H) -> Node[H]:
        return self.nodes[key]

    def add_edge(self, k_u: H, k_v: H):
        logger.trace(f"Adding edge {k_u} -> {k_v}")
        u = self.nodes.setdefault(k_u, self.node_class(k_u))
        v = self.nodes.setdefault(k_v, self.node_class(k_v))
        u.add_neighbor(v)

    def reset_nodes(self):
        for node in self.nodes.values():
            node.reset_node()

    def reset(self):
        for node in self.nodes.values():
            node.reset_node()
        self.nodes.clear()

    def __iter__(self) -> Iterator[Node[H]]:
        return iter(sorted(self.nodes.values(), key=lambda node: node.key))

    def transpose(self) -> Graph[H]:
        graph = Graph(node_class=self.node_class)
        for node in self.nodes.values():
            for neighbor in node.adj:
                graph.add_edge(neighbor.key, node.key)
        return graph

    @staticmethod
    def ascii_namer(i: int) -> str:
        t = i
        digits = []
        while True:
            t, r = divmod(t, 26)
            digits.append(r)
            if t == 0:
                break
        return "".join(chr(ord("a") + d) for d in reversed(digits))

    def __hash__(self) -> int:
        vertices = tuple(sorted(self.nodes.keys()))
        edges = set()
        for node in self.nodes.values():
            for neighbor in node.adj:
                edges.add((node.key, neighbor.key))
        edges_tuple = tuple(sorted(edges))
        return hash((vertices, edges_tuple))

    @classmethod
    def random(
        cls,
        *,
        n: int,
        p: float,
        seed: int = 42,
        namer: Callable[[int], H] = lambda i: i,
    ) -> Graph[H]:
        rand = rn.Random(seed)
        graph = cls(node_class=Node2)
        for i in range(n):
            for j in range(i + 1, n):
                if rand.random() < p:
                    graph.add_edge(namer(i), namer(j))
        return graph
