#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.pytest -p python313Packages.click
from __future__ import annotations

from collections.abc import MutableSequence

import click
import pytest


def solution(xs: MutableSequence[int]) -> None:
    """
    N-way Dutch National Flag partitioning with O(1) space and O(nk) time.

    This algorithm partitions an array containing k distinct values into k groups,
    where all instances of the same value are grouped together in sorted order.

    Time complexity: O(nk) where n = len(xs) and k = number of unique values
    Space complexity: O(1) - only uses a constant number of pointer variables

    The algorithm works by repeatedly:
    1. Finding the minimum value in the remaining unsorted portion
    2. Partitioning all occurrences of that minimum to the left
    3. Moving the start pointer past the partitioned elements
    """
    n = len(xs)
    if n <= 1:
        return

    start = 0
    while start < n:
        min_val = xs[start]
        for i in range(start + 1, n):
            # We cannot pre-compute and store the next minimum without using
            # O(k) space. So we must scan in worst case O(n) time.
            if xs[i] < min_val:
                min_val = xs[i]

        left = start
        current = start
        while current < n:
            if xs[current] == min_val:
                if current != left:
                    xs[left], xs[current] = xs[current], xs[left]
                left += 1
            current += 1

        start = left if left > start else start + 1


@pytest.mark.parametrize(
    "input_array, expected",
    [
        (
            [1, 0, 1, 0, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2],
        ),
        (
            [2, 2, 3, 5, 5, 5, 1, 1, 4, 4, 5, 5, 5, 3, 3, 5, 1, 2, 2, 5],
            [1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5],
        ),
        ([42], [42]),
        ([7, 7, 7, 7, 7], [7, 7, 7, 7, 7]),
        ([1, 1, 2, 2, 3, 3, 4, 4], [1, 1, 2, 2, 3, 3, 4, 4]),
        ([5, 5, 4, 4, 3, 3, 2, 2, 1, 1], [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]),
        ([1, 0, 1, 1, 0, 0, 1, 0, 1, 0], [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
        (
            [10, 3, 7, 3, 10, 1, 7, 1, 3, 7, 10, 1, 3],
            [1, 1, 1, 3, 3, 3, 3, 7, 7, 7, 10, 10, 10],
        ),
        ([], []),
    ],
)
def test_solution(input_array: list[int], expected: list[int]) -> None:
    solution(input_array)
    assert input_array == expected


cli = click.Group()


@cli.command()
@click.argument("numbers", nargs=-1, type=int)
def partition(numbers: tuple[int, ...]):
    """Partition an array of numbers using n-way Dutch flag algorithm."""
    xs = list(numbers)
    click.echo(f"Before: {xs}")
    solution(xs)
    click.echo(f"After:  {xs}")


@cli.command("test")
def run_tests():
    """Run pytest tests."""
    pytest.main([__file__])


if __name__ == "__main__":
    cli()
