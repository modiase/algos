#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.pytest -p python313Packages.click
from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Literal, assert_never

import click
import pytest


class UnionFind:
    def __init__(self, n):
        self._parents = list(range(n))
        self._rank = [1] * n

    def find(self, e: int) -> int:
        result = self._parents[e]
        while result != self._parents[result]:
            self._parents[result] = self._parents[
                self._parents[result]
            ]  # path compression
            result = self._parents[result]

        return result

    def union(self, e1: int, e2: int) -> bool:
        if e1 > e2:
            return self.union(e2, e1)

        if self.find(e1) == self.find(e2):
            return False

        self._parents[e2] = self._parents[e1]
        self._rank[e1] += self._rank[e2]
        return True


def countConnectedComponents(edges: Collection[tuple[int, int]], n: int) -> int:
    djs = UnionFind(n)
    components = n
    for v1, v2 in edges:
        components -= int(djs.union(v1, v2))

    return components


@pytest.mark.parametrize(
    "edges, n, expected",
    [
        ([(0, 1), (1, 2), (2, 0)], 3, 1),
        ([(0, 1), (2, 3)], 4, 2),
        ([(0, 1), (1, 2), (3, 4)], 5, 2),
        ([], 5, 5),
        ([(0, 1)], 2, 1),
        ([(0, 1), (0, 2), (0, 3), (0, 4)], 5, 1),
        ([(0, 1), (2, 3), (4, 5)], 6, 3),
        ([(0, 1), (1, 2), (2, 3), (3, 4)], 5, 1),
        ([], 1, 1),
        ([(0, 0)], 1, 1),
    ],
)
def test_count_connected_components(
    edges: list[tuple[int, int]], n: int, expected: int
) -> None:
    assert countConnectedComponents(edges, n) == expected


type Operation = tuple[Literal["find"], int] | tuple[Literal["union"], int, int]


@pytest.mark.parametrize(
    "n, operations, expected",
    [
        (
            5,
            [("union", 0, 1), ("union", 1, 2), ("find", 0), ("find", 2)],
            [True, True, 0, 0],
        ),
        (
            3,
            [("union", 0, 1), ("union", 1, 2), ("find", 0), ("find", 2)],
            [True, True, 0, 0],
        ),
        (
            4,
            [("union", 0, 1), ("union", 2, 3), ("find", 0), ("find", 2)],
            [True, True, 0, 2],
        ),
        (2, [("union", 0, 1), ("union", 0, 1)], [True, False]),
    ],
)
def test_union_find(
    n: int, operations: Sequence[Operation], expected: list[bool | int]
) -> None:
    uf = UnionFind(n)
    results = []
    for op in operations:
        match op:
            case ("union", op1, op2):
                results.append(uf.union(op1, op2))
            case ("find", op):
                results.append(uf.find(op))
            case never:
                assert_never(never)
    assert results == expected


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option("--n", default=5, help="Number of nodes")
@click.option("--edges", default="0,1;1,2;3,4", help="Edges as 'u1,v1;u2,v2;...'")
def count(n: int, edges: str) -> None:
    """Count connected components in an undirected graph."""

    def parse_edge(edge_str: str) -> tuple[int, int]:
        u, v = edge_str.split(",")
        return (int(u), int(v))

    edge_list = [parse_edge(e) for e in edges.split(";")] if edges else []
    components = countConnectedComponents(edge_list, n)
    click.echo(f"Number of connected components: {components}")


@cli.command("test")
def run_tests() -> None:
    """Run all tests."""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    cli()
