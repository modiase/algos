#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.loguru -p python313Packages.pytest -p python313Packages.click
from __future__ import annotations

import random
from collections.abc import Sequence

import click
import pytest


def sortedness_by_inversions(arr: Sequence[int]) -> float:
    n = len(arr)
    if n <= 1:
        return 0.0

    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if arr[i] > arr[j]:
                count += 1

    return count / (n * (n - 1) // 2)


def sortedness_by_lis(arr: Sequence[int]) -> float:
    if not arr:
        return 1.0

    n = len(arr)
    dp = [1] * n

    for i in range(1, n):
        for j in range(i):
            if arr[j] < arr[i]:
                dp[i] = max(dp[i], dp[j] + 1)

    return max(dp) / n


@pytest.mark.parametrize(
    "arr, expected",
    [
        ([1, 2, 3, 4], 0.0),
        ([4, 3, 2, 1], 1.0),
        ([2, 1, 3, 4], 1 / 6),
        ([1, 3, 2, 4], 1 / 6),
        ([3, 1, 2], 2 / 3),
        ([], 0.0),
        ([5], 0.0),
    ],
)
def test_sortedness_by_inversions(arr: Sequence[int], expected: float) -> None:
    assert abs(sortedness_by_inversions(arr) - expected) < 1e-9


@pytest.mark.parametrize(
    "arr, expected",
    [
        ([1, 2, 3, 4], 1.0),
        ([4, 3, 2, 1], 0.25),
        ([1, 3, 2, 4], 0.75),
        ([2, 1, 3, 4], 0.75),
        ([], 1.0),
    ],
)
def test_sortedness_by_lis(arr: Sequence[int], expected: float) -> None:
    assert sortedness_by_lis(arr) == expected


cli = click.Group()


@cli.command()
@click.option("--seed", "-s", default=42, help="Random seed for array generation")
@click.option("--n", "-n", default=10, help="Size of the array to generate")
def demo(seed: int, n: int) -> None:
    random.seed(seed)
    arr = list(range(1, n + 1))
    random.shuffle(arr)

    click.echo(f"Array (seed={seed}, N={n}): {arr}")
    click.echo(f"Sortedness by inversions: {sortedness_by_inversions(arr):.3f}")
    click.echo(f"Sortedness by LIS: {sortedness_by_lis(arr):.3f}")


@cli.command("test")
def run_tests() -> None:
    pytest.main([__file__])


if __name__ == "__main__":
    cli()
