#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.pytest -p python313Packages.click -p python313Packages.loguru
from __future__ import annotations

import random
from collections.abc import Sequence

import click
import pytest
from loguru import logger


def kadane(arr: Sequence[int]) -> tuple[int, int, int, int]:
    """
    Kadane's algorithm for maximum subarray sum.
    Returns (max_sum, start_idx, end_idx, current_sum).
    Time complexity: O(n)
    Space complexity: O(1)
    """
    if not arr:
        return 0, 0, -1, 0

    max_sum = arr[0]
    max_start = 0
    max_end = 0
    current_sum = arr[0]
    current_start = 0

    logger.trace(f"Initial: arr[0]={arr[0]}, max_sum={max_sum}")

    for i in range(1, len(arr)):
        if current_sum < 0:
            current_sum = arr[i]
            current_start = i
        else:
            current_sum += arr[i]

        logger.trace(
            f"i={i}, arr[i]={arr[i]}, current_sum={current_sum}, "
            f"current_start={current_start}, max_sum={max_sum}"
        )

        if current_sum > max_sum:
            max_sum = current_sum
            max_start = current_start
            max_end = i
            logger.trace(
                f"  New max! max_sum={max_sum}, range=[{max_start}:{max_end + 1}]"
            )

    return max_sum, max_start, max_end, current_sum


def max_subarray_divide_conquer(
    arr: Sequence[int], low: int = 0, high: int | None = None
) -> tuple[int, int, int]:
    """
    Divide and conquer approach to maximum subarray (CLRS Chapter 4).
    Returns (max_sum, start_idx, end_idx).
    Time complexity: O(n log n)
    Space complexity: O(log n) for recursion stack
    """
    if high is None:
        high = len(arr) - 1

    if low == high:
        return arr[low], low, high

    mid = (low + high) // 2
    left_sum, left_start, left_end = max_subarray_divide_conquer(arr, low, mid)
    right_sum, right_start, right_end = max_subarray_divide_conquer(arr, mid + 1, high)
    cross_sum, cross_start, cross_end = max_crossing_subarray(arr, low, mid, high)

    logger.trace(
        f"Range [{low}:{high + 1}]: left={left_sum}, right={right_sum}, cross={cross_sum}"
    )

    if left_sum >= right_sum and left_sum >= cross_sum:
        return left_sum, left_start, left_end
    elif right_sum >= left_sum and right_sum >= cross_sum:
        return right_sum, right_start, right_end
    else:
        return cross_sum, cross_start, cross_end


def max_crossing_subarray(
    arr: Sequence[int], low: int, mid: int, high: int
) -> tuple[int, int, int]:
    """
    Find maximum subarray crossing the midpoint.
    Helper for divide and conquer approach.
    """
    left_sum = float("-inf")
    current_sum = 0
    max_left = mid

    for i in range(mid, low - 1, -1):
        current_sum += arr[i]
        if current_sum > left_sum:
            left_sum = current_sum
            max_left = i

    right_sum = float("-inf")
    current_sum = 0
    max_right = mid + 1

    for i in range(mid + 1, high + 1):
        current_sum += arr[i]
        if current_sum > right_sum:
            right_sum = current_sum
            max_right = i

    return left_sum + right_sum, max_left, max_right


@pytest.mark.parametrize(
    "arr, expected_sum",
    [
        ([-2, 1, -3, 4, -1, 2, 1, -5, 4], 6),
        ([1], 1),
        ([5, 4, -1, 7, 8], 23),
        ([-1, -2, -3, -4], -1),
        ([1, 2, 3, 4, 5], 15),
        ([-2, -1], -1),
        ([2, -1, 2, 3, 4, -5], 10),
        ([], 0),
    ],
)
def test_kadane(arr: Sequence[int], expected_sum: int) -> None:
    max_sum, start, end, _ = kadane(arr)
    assert max_sum == expected_sum
    if arr:
        assert sum(arr[start : end + 1]) == expected_sum


@pytest.mark.parametrize(
    "arr, expected_sum",
    [
        ([-2, 1, -3, 4, -1, 2, 1, -5, 4], 6),
        ([1], 1),
        ([5, 4, -1, 7, 8], 23),
        ([-1, -2, -3, -4], -1),
        ([1, 2, 3, 4, 5], 15),
        ([-2, -1], -1),
        ([2, -1, 2, 3, 4, -5], 10),
    ],
)
def test_divide_conquer(arr: Sequence[int], expected_sum: int) -> None:
    max_sum, start, end = max_subarray_divide_conquer(arr)
    assert max_sum == expected_sum
    assert sum(arr[start : end + 1]) == expected_sum


def test_both_algorithms_agree() -> None:
    test_cases = [
        [-2, 1, -3, 4, -1, 2, 1, -5, 4],
        [1, 2, 3, 4, 5],
        [-1, -2, -3, -4],
        [5, 4, -1, 7, 8],
        list(range(-10, 10)),
        list(range(10, -10, -1)),
    ]

    for arr in test_cases:
        kadane_sum, _, _, _ = kadane(arr)
        dc_sum, _, _ = max_subarray_divide_conquer(arr)
        assert kadane_sum == dc_sum, (
            f"Mismatch for {arr}: Kadane={kadane_sum}, DC={dc_sum}"
        )


@click.group()
@click.option(
    "--log-level",
    "-l",
    default="INFO",
    type=click.Choice(
        ["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR"], case_sensitive=False
    ),
    help="Set logging level",
)
@click.pass_context
def cli(ctx: click.Context, log_level: str) -> None:
    ctx.ensure_object(dict)
    ctx.obj["log_level"] = log_level.upper()


@cli.command()
@click.option("--size", "-n", default=20, help="Size of array")
@click.option("--seed", "-s", default=42, help="Random seed")
@click.option("--min-value", default=-10, help="Minimum value in array")
@click.option("--max-value", default=10, help="Maximum value in array")
@click.pass_context
def demo(
    ctx: click.Context, size: int, seed: int, min_value: int, max_value: int
) -> None:
    logger.remove()
    logger.add(
        lambda msg: click.echo(msg, err=True), level=ctx.obj["log_level"], colorize=True
    )

    logger.info("=" * 60)
    logger.info("MAXIMUM SUBARRAY PROBLEM")
    logger.info("=" * 60)

    rng = random.Random(seed)
    arr = [rng.randint(min_value, max_value) for _ in range(size)]

    click.echo(f"\nArray: {arr}")
    click.echo(f"Length: {len(arr)}")

    logger.info("\nKadane's Algorithm (O(n)):")
    max_sum_k, start_k, end_k, _ = kadane(arr)
    click.echo(f"  Maximum sum: {max_sum_k}")
    click.echo(f"  Subarray: {arr[start_k : end_k + 1]}")
    click.echo(f"  Indices: [{start_k}:{end_k + 1}]")

    logger.info("\nDivide and Conquer (O(n log n)):")
    max_sum_dc, start_dc, end_dc = max_subarray_divide_conquer(arr)
    click.echo(f"  Maximum sum: {max_sum_dc}")
    click.echo(f"  Subarray: {arr[start_dc : end_dc + 1]}")
    click.echo(f"  Indices: [{start_dc}:{end_dc + 1}]")

    if max_sum_k == max_sum_dc:
        logger.success(f"\n✓ Both algorithms agree: Maximum sum = {max_sum_k}")
    else:
        logger.error(f"\n✗ Mismatch: Kadane={max_sum_k}, DC={max_sum_dc}")


@cli.command()
@click.argument("values", nargs=-1, type=int)
@click.pass_context
def solve(ctx: click.Context, values: tuple[int, ...]) -> None:
    logger.remove()
    logger.add(
        lambda msg: click.echo(msg, err=True), level=ctx.obj["log_level"], colorize=True
    )

    if not values:
        click.echo("Error: Please provide at least one value")
        return

    arr = list(values)

    click.echo(f"Array: {arr}")

    max_sum, start, end, _ = kadane(arr)

    click.echo(f"Maximum sum: {max_sum}")
    click.echo(f"Subarray: {arr[start : end + 1]}")
    click.echo(f"Indices: [{start}:{end + 1}]")


@cli.command()
def test() -> None:
    logger.disable("")
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    cli()
