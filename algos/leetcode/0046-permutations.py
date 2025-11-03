#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.pytest -p python313Packages.click
from __future__ import annotations

from collections.abc import Collection, Sequence

import click
import pytest


def permute(nums: Sequence[int]) -> Collection[tuple[int, ...]]:
    N = len(nums)

    def _permute(nums, j):
        if j == N - 1:
            yield nums[:]
        else:
            for i in range(j, N):
                nums[i], nums[j] = nums[j], nums[i]
                yield from _permute(nums, j + 1)
                nums[i], nums[j] = nums[j], nums[i]

    return set(tuple(p) for p in _permute(nums, 0))


@pytest.mark.parametrize(
    "nums, expected",
    [
        ([1, 2, 3], {(1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1)}),
        ([0, 1], {(0, 1), (1, 0)}),
        ([1], {(1,)}),
        ([], set()),
        ([1, 2], {(1, 2), (2, 1)}),
        ([1, 1, 2], {(1, 1, 2), (1, 2, 1), (2, 1, 1)}),
    ],
)
def test_permute(nums: list[int], expected: set[tuple[int, ...]]) -> None:
    assert permute(nums) == expected


def test_permute_length() -> None:
    import math

    for n in range(1, 6):
        nums = list(range(n))
        result = permute(nums)
        assert len(result) == math.factorial(n)


def test_permute_all_elements_present() -> None:
    nums = [1, 2, 3]
    result = permute(nums)
    for perm in result:
        assert set(perm) == set(nums)
        assert len(perm) == len(nums)


cli = click.Group()


@cli.command()
@click.argument("numbers", nargs=-1, type=int)
def generate(numbers: tuple[int, ...]) -> None:
    """Generate all permutations of the given numbers."""
    if not numbers:
        click.echo("Please provide numbers to permute")
        return

    perms = permute(list(numbers))
    click.echo(f"Found {len(perms)} unique permutations:")
    for perm in sorted(perms):
        click.echo(perm)


@cli.command("test")
def run_tests() -> None:
    """Run all tests."""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    cli()
