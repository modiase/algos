from collections.abc import Hashable, Iterator, Mapping
from typing import TypeVar

from loguru import logger
from more_itertools import zip_broadcast

simple_graph = {"A": {"B": 1, "C": 4}, "B": {"C": 2, "D": 7}, "C": {"D": 3}, "D": {}}
simple_graph_with_negative_cycle = {
    "A": {"B": 1, "C": 4},
    "B": {"C": 2, "D": 7},
    "C": {"D": 3},
    "D": {"A": -10},
}
simple_graph_with_positive_cycle = {
    "A": {"B": 1, "C": 4},
    "B": {"C": 2, "D": 7},
    "C": {"D": 3},
    "D": {"A": 10},
}
hub_graph = {
    "A": {"B": 1, "C": 4, "D": 10},
    "B": {},
    "C": {},
    "D": {},
}
disconnected_graph = {"A": {}, "B": {}, "C": {}, "D": {}}

_T = TypeVar("_T", bound=Hashable)
Graph = Mapping[_T, Mapping[_T, float]]
Edge = tuple[_T, _T, float]


def graph_edge_iterator(g: Graph) -> Iterator[Edge]:
    yield from (
        (node, neighbour, w)
        for node, neighbours_to_weight in g.items()
        for node, (neighbour, w) in zip_broadcast(node, neighbours_to_weight.items())
    )


def bellman_ford(graph: Graph, start: _T) -> Mapping[_T, float] | None:
    distances = {
        node: graph[start].get(node, float("inf")) if node != start else 0
        for node in graph
    }

    for i in range(len(graph) - 1):
        flag = False
        for node, neighbour, weight in graph_edge_iterator(graph):
            if distances[neighbour] > distances[node] + weight:
                distances[neighbour] = distances[node] + weight
                flag = True
        if not flag:
            logger.debug("Early exit on iteration {}", i)
            break

    for node, neighbour, weight in graph_edge_iterator(graph):
        if distances[neighbour] > distances[node] + weight:
            return None

    return distances


if __name__ == "__main__":
    logger.info(f"Simple graph: {bellman_ford(simple_graph, 'A')}")
    logger.info(
        f"Simple graph with negative cycle: {bellman_ford(simple_graph_with_negative_cycle, 'A')}"
    )
    logger.info(
        f"Simple graph with positive cycle: {bellman_ford(simple_graph_with_positive_cycle, 'A')}"
    )
    logger.info(f"Hub graph: {bellman_ford(hub_graph, 'A')}")
    logger.info(f"Disconnected graph: {bellman_ford(disconnected_graph, 'A')}")
