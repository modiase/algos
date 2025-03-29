"""
Suppose you are given a table of currency exchange rates, represented as a 2D
array.  Determine whether there is a possible arbitrage: that is, whether there
is some sequence of trades you can make, starting with some amount A of any
currency, so that you can end up with some amount greater than A of that
currency.

There are no transaction costs and you can trade fractional quantities.
"""

import math
from collections.abc import Collection, Hashable, Iterator, Mapping, Sequence
from csv import DictReader
from decimal import Decimal, getcontext
from itertools import chain, islice
from pathlib import Path
from typing import TypeVar

from loguru import logger
from more_itertools import pairwise, unique, zip_broadcast

_K = TypeVar("_K", bound=Hashable)
_N = TypeVar("_N", float, Decimal)
Graph = Mapping[_K, Mapping[_K, _N]]
Edge = tuple[_K, _K, _N]

getcontext().prec = 30


def graph_edge_iterator(g: Graph) -> Iterator[Edge]:
    yield from (
        (node, neighbour, w)
        for node, neighbours_to_weight in g.items()
        for node, (neighbour, w) in zip_broadcast(node, neighbours_to_weight.items())
    )


def bellman_ford(graph: Graph, start: _K) -> Mapping[_K, float] | None:
    distances = {
        node: graph[start].get(node, float("inf")) if node != start else 0
        for node in graph
    }

    for _ in range(len(graph) - 1):
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


def find_arbitrages_for_start(
    negative_log_rates: Graph, start: _K
) -> Iterator[tuple[Sequence[_K], Decimal]]:
    stack = [(start, ())]
    while stack:
        # TODO: consider how we could reduce the size of the search space since
        # there will be a lot of duplicate paths.
        node, path = stack.pop()
        for neighbour in negative_log_rates[node]:
            if neighbour in path:
                continue
            if neighbour == start and (
                path_sum := sum(
                    negative_log_rates[c][c_next]
                    for c, c_next in pairwise((start, *path, start))
                )
            ) < Decimal(0):
                yield ((start, *path, start), math.exp(-path_sum))
                continue
            if neighbour != start and neighbour not in path:
                stack.append((neighbour, (*path, neighbour)))


def find_arbitrages(
    rates: Mapping[_K, Mapping[_K, Decimal]],
    find_all: bool = False,
) -> Collection[tuple[Sequence[_K], Decimal]]:
    negative_log_rates = {
        currency: {
            neighbour: Decimal(-math.log(rate))
            for neighbour, rate in rates[currency].items()
        }
        for currency in rates
    }
    return list(
        unique(
            islice(
                chain.from_iterable(
                    (
                        find_arbitrages_for_start(
                            (
                                negative_log_rates := {
                                    currency: {
                                        neighbour: Decimal(-math.log(rate))
                                        for neighbour, rate in rates[currency].items()
                                    }
                                    for currency in rates
                                }
                            ),
                            start_currency,
                        )
                        for start_currency in {
                            currency
                            for currency in negative_log_rates
                            if bellman_ford(negative_log_rates, currency) is None
                        }
                    ),
                ),
                None if find_all else 1,
            ),
            key=lambda x: sorted(x[0][:-1]),
        )
    )


def load_rates(path: Path) -> Mapping[str, Mapping[str, Decimal]]:
    with open(path, "r") as f:
        return {
            (currency := row.pop("Currency")): {
                k: Decimal(v) for k, v in row.items() if k != currency
            }
            for row in DictReader(f)
        }


if __name__ == "__main__":
    logger.info(
        find_arbitrages(load_rates(Path(__file__).parent / "rates.csv"), find_all=True)
    )
    logger.info(
        find_arbitrages(load_rates(Path(__file__).parent / "rates2.csv"), find_all=True)
    )
    logger.info(
        find_arbitrages(load_rates(Path(__file__).parent / "rates3.csv"), find_all=True)
    )
