from __future__ import annotations

import random as rn
from collections.abc import Collection, Hashable, Iterator, MutableMapping
from enum import IntEnum
from typing import Callable, Generic, TypeVar, cast

from loguru import logger

H = TypeVar("H", bound=Hashable)


class AdjacencyList(Generic[H]):
    def __init__(self, sort_key: Callable[[Node[H]], float] | None = None):
        self.nodes: set[Node[H]] = set()
        self.weights: MutableMapping[Node[H], float] = {}
        self.sort_key = sort_key

    def add(self, node: Node[H], weight: float):
        self.nodes.add(node)
        self.weights[node] = weight

    def get_weight(self, node: Node[H]) -> float:
        return self.weights.get(node, float("inf"))

    def __iter__(self) -> Iterator[tuple[Node[H], float]]:
        gen = ((n, self.weights[n]) for n in self.nodes)
        if self.sort_key is None:
            return iter(gen)
        # mypy doesn't understand that sort_key is not None here
        sort_key = self.sort_key
        return iter(sorted(gen, key=lambda item: sort_key(item[0])))

    def __repr__(self) -> str:
        return f"AdjacencyList({', '.join(f'{node.key}({weight})' for node, weight in self)})"

    def __str__(self) -> str:
        return f"{', '.join(f'{node.key}({weight})' for node, weight in self)}"

    def __contains__(self, node: Node[H]) -> bool:
        return node in self.nodes

    def __len__(self) -> int:
        return len(self.nodes)


class Node(Generic[H]):
    def __init__(self, key: H, sort_key: Callable[[Node[H]], float] | None = None):
        self.key = key
        self.adj: AdjacencyList[H] = AdjacencyList(sort_key=sort_key)
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
    def predecessor(self, predecessor: Node[H] | None):
        logger.trace(
            f"Setting predecessor of {self.key} to {predecessor.key if predecessor else None}"
        )
        self._predecessor = predecessor

    @property
    def start_time(self) -> int | None:
        return self._start_time

    @start_time.setter
    def start_time(self, time: int | None):
        logger.trace(f"Setting start time of {self.key} to {time}")
        self._start_time = time

    @property
    def end_time(self) -> int | None:
        return self._end_time

    @end_time.setter
    def end_time(self, time: int | None):
        logger.trace(f"Setting end time of {self.key} to {time}")
        self._end_time = time

    def __eq__(self, other: object):
        if isinstance(other, Node):
            return self.key == other.key
        elif isinstance(other, type(self.key)):
            return bool(self.key == other)
        return NotImplemented

    def __hash__(self):
        return hash(self.key)

    def add_neighbor(self, neighbor: Node[H], weight: float):
        logger.trace(
            f"Adding neighbor {neighbor.key} with weight {weight} to {self.key}"
        )
        self.adj.add(neighbor, weight)

    def __str__(self):
        return f"Node({self.key})"

    def __repr__(self):
        return f"Node({self.key}, adj=[{', '.join(f'{node.key}({weight})' for node, weight in self.adj)}])"

    def reset_node(self):
        self.start_time = None
        self.end_time = None
        self.predecessor = None

    @property
    def edges(self) -> Iterator[tuple[Node[H], float]]:
        return iter(self.adj)


class Node2(Node[H]):
    class Color(IntEnum):
        WHITE = 0  # not visited
        GRAY = 1  # being visited
        BLACK = 2  # visited

    def __init__(self, key: H, sort_key: Callable[[Node[H]], float] | None = None):
        super().__init__(key, sort_key=sort_key)
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
        *,
        nodes: Collection[H] | None = None,
        edges: Collection[tuple[H, H, float]] | None = None,
        node_class: type[Node[H]] = Node,
        sort_key: Callable[[Node[H]], float] | None = None,
    ):
        self._sort_key = sort_key
        self.nodes: dict[H, Node[H]] = {}
        self.node_class = node_class
        self.sort_key = sort_key
        if nodes is not None:
            for node in nodes:
                self.add_node(node)
        for k_u, k_v, weight in edges or []:
            self.add_edge(k_u, k_v, weight)

    def add_node(self, key: H):
        if key not in self.nodes:
            self.nodes[key] = self.node_class(key, sort_key=self.sort_key)

    def __getitem__(self, key: H) -> Node[H]:
        return self.nodes[key]

    def add_edge(self, k_u: H, k_v: H, weight: float):
        logger.trace(f"Adding edge {k_u} -> {k_v} with weight {weight}")
        u = self.nodes.setdefault(k_u, self.node_class(k_u, sort_key=self.sort_key))
        v = self.nodes.setdefault(k_v, self.node_class(k_v, sort_key=self.sort_key))
        u.add_neighbor(v, weight)

    def reset_nodes(self):
        for node in self.nodes.values():
            node.reset_node()

    def reset(self):
        for node in self.nodes.values():
            node.reset_node()
        self.nodes.clear()

    def __iter__(self) -> Iterator[Node[H]]:
        if self._sort_key is None:
            return iter(self.nodes.values())
        return iter(sorted(self.nodes.values(), key=self._sort_key))

    def transpose(self) -> Graph[H]:
        graph = Graph(node_class=self.node_class)
        for node in self.nodes.values():
            graph.add_node(node.key)
            for neighbor, weight in node.adj:
                graph.add_edge(neighbor.key, node.key, weight)
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
        vertices = tuple(sorted(self.nodes.keys(), key=str))
        edges = set()
        for node in self.nodes.values():
            for neighbor, _ in node.adj:
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
        index_to_name: Callable[[int], H] | None = None,
    ) -> Graph[H]:
        def default_namer(i: int) -> H:
            return cast(H, i)

        namer_func: Callable[[int], H]
        if index_to_name is None:
            namer_func = default_namer
        else:
            namer_func = index_to_name
        rand = rn.Random(seed)
        graph = cls(node_class=Node2)
        for i in range(n):
            for j in range(i + 1, n):
                if rand.random() < p:
                    graph.add_edge(namer_func(i), namer_func(j), rn.uniform(0.1, 10.0))
                if rand.random() < p:
                    graph.add_edge(namer_func(j), namer_func(i), rn.uniform(0.1, 10.0))
        return graph

    def __repr__(self) -> str:
        return (
            f"Graph({{nodes={', '.join(repr(node) for node in self.nodes.values())})}}"
        )

    def __len__(self) -> int:
        return len(self.nodes)

    @property
    def edges(self) -> Iterator[tuple[Node[H], Node[H], float]]:
        yield from (
            (node, neighbor, weight)
            for node in self.nodes.values()
            for neighbor, weight in iter(node.adj)
        )
