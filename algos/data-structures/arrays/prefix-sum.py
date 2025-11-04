#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.pytest -p python313Packages.click
from __future__ import annotations

from collections.abc import Iterable

import click
import pytest


class PrefixSumArray:
    def __init__(self, it: Iterable[int]) -> None:
        self._array = []

        iterator = iter(it)
        head = next(iterator, None)
        if head is not None:
            self._array.append(head)
            for v in iterator:
                self._array.append(self._array[-1] + v)

    def query(self, start: int, end: int) -> int:
        if not self._array or end <= start or end > len(self._array):
            return 0
        end_sum = self._array[end - 1]
        start_sum = self._array[start - 1] if start > 0 else 0
        return end_sum - start_sum


@pytest.mark.parametrize(
    "input_array, queries, expected_results",
    [
        ([1, 2, 3, 4, 5], [(0, 2), (1, 3), (0, 4), (0, 5)], [3, 5, 10, 15]),
        (
            [10, 20, 30, 40],
            [(0, 0), (0, 1), (1, 2), (0, 3), (2, 4)],
            [0, 10, 20, 60, 70],
        ),
        ([5], [(0, 0), (0, 1)], [0, 5]),
        ([-1, 2, -3, 4, -5], [(0, 1), (2, 4), (0, 4), (1, 5)], [-1, 1, 2, -2]),
        ([1, 1, 1, 1, 1], [(0, 4), (1, 3), (2, 2), (0, 5)], [4, 2, 0, 5]),
    ],
)
def test_prefix_sum_array(
    input_array: list[int], queries: list[tuple[int, int]], expected_results: list[int]
) -> None:
    psa = PrefixSumArray(iter(input_array))
    for (start, end), expected in zip(queries, expected_results):
        assert psa.query(start, end) == expected


def test_prefix_sum_empty() -> None:
    psa = PrefixSumArray(iter([]))
    assert psa.query(0, 0) == 0


cli = click.Group()


@cli.command()
@click.argument("numbers", nargs=-1, type=int)
def build(numbers: tuple[int, ...]):
    """Build a prefix sum array and show its contents."""
    psa = PrefixSumArray(iter(numbers))
    click.echo(f"Input: {list(numbers)}")
    click.echo(f"Prefix sum array: {psa._array}")


@cli.command()
@click.argument("numbers", nargs=-1, type=int)
def query(numbers: tuple[int, ...]):
    """Build prefix sum and perform interactive queries."""
    if len(numbers) < 2:
        click.echo("Usage: query <array_elements...>")
        return

    psa = PrefixSumArray(iter(numbers))
    click.echo(f"Input array: {list(numbers)}")
    click.echo(f"Prefix sum array: {psa._array}")
    click.echo("\nExample queries:")
    click.echo(f"  Range [0, 2]: {psa.query(0, 2)}")
    if len(numbers) > 3:
        click.echo(f"  Range [1, 3]: {psa.query(1, 3)}")
    click.echo(f"  Range [0, {len(numbers) - 1}]: {psa.query(0, len(numbers) - 1)}")


@cli.command("test")
def run_tests():
    """Run pytest tests."""
    pytest.main([__file__])


if __name__ == "__main__":
    cli()
