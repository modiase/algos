"""
Suppose you are given a table of currency exchange rates, represented as a 2D array.
Determine whether there is a possible arbitrage: that is, whether there is some sequence of trades you can make, starting with some amount
A of any currency, so that you can end up with some amount greater than A of that currency.

There are no transaction costs and you can trade fractional quantities.
"""

import math
from collections.abc import Hashable, Iterator, Mapping, Sequence
from csv import DictReader
from pathlib import Path
from typing import TypeVar

from more_itertools import zip_broadcast

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
            break

    for node, neighbour, weight in graph_edge_iterator(graph):
        if distances[neighbour] > distances[node] + weight:
            return None

    return distances

def find_arbitrage(negative_log_rates: Graph, start: _T) -> Sequence[_T]:
    stack = [(start, (), 0)]
    while stack:
        node, path, weight = stack.pop()
        for neighbour, w in negative_log_rates[node].items():
            if neighbour == start:
                return (neighbour, *path)
            stack.append((neighbour, (neighbour, *path), weight + w))


def exists_arbitrage(
    rates: Mapping[str, Mapping[str, float]],
) -> Sequence[Sequence[_T]]:
    negative_log_rates = {
        currency: {
            neighbour: -math.log(rate) for neighbour, rate in rates[currency].items()
        }
        for currency in rates
    }

    cycle_starts = set()
    for currency in rates:
        if bellman_ford(negative_log_rates, currency) is None:
            cycle_starts.add(currency)

    cycles = []
    while len(cycle_starts) > 0:
        start = cycle_starts.pop()
        cycle = find_arbitrage(negative_log_rates, start)
        for node in cycle:
            if node in cycle_starts:
                cycle_starts.remove(node)
        cycles.append(cycle)

    return cycles


def load_rates(path: Path) -> Mapping[str, Mapping[str, float]]:
    with open(path, "r") as f:
        return {
            (currency := row.pop("Currency")): {
                k: float(v) for k, v in row.items() if k != currency
            }
            for row in DictReader(f)
        }


if __name__ == "__main__":
    # TODO: Add tests
    print(exists_arbitrage(load_rates(Path(__file__).parent / "rates.csv")))
    print(exists_arbitrage(load_rates(Path(__file__).parent / "rates2.csv")))
