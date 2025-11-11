#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.loguru -p python313Packages.pytest -p python313Packages.pyvis -p python313Packages.more-itertools -p python313Packages.click
"""
Suffix array construction using the doubling algorithm.

The key insight: rank[i] represents increasingly longer substrings as we iterate.
- Iteration 1: rank[i] = ordering of the 1-character substring at position i
- Iteration 2: rank[i] = ordering of the 2-character substring at position i
- Iteration 3: rank[i] = ordering of the 4-character substring at position i

When we sort by (rank[i], rank[i+k]), we construct the rank of a 2k-character block
by comparing two k-character blocks: rank[i] for positions [i, i+k) and rank[i+k]
for positions [i+k, i+2k). The ranks from iteration n become the building blocks for
iteration n+1, encoding all previous comparison results without re-examining characters.

Example with "banana": After iteration 1, "an" has rank 1 and "a" has rank 0.
In iteration 2, "anan" gets key (1, 1) because rank("an") + rank("an") = (1, 1).
Similarly, "ana" gets key (1, 0) because rank("an") + rank("a") = (1, 0).
Since (1, 0) < (1, 1), we correctly determine "ana" < "anan" without re-comparing.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import click
import pytest
from loguru import logger


def radix_sort_pairs(
    indices: list[int], get_key: Callable[[int], tuple[int, int]], max_value: int
) -> list[int]:
    """
    Sort indices by integer pair keys using radix sort (counting sort).
    """
    n = len(indices)
    logger.trace(f"Radix sort: n={n}, max_value={max_value}")

    logger.trace("  Pass 1: sorting by second component")
    # Second component ranges from -1 to max_value, so shift by +1
    count = [0] * (max_value + 2)

    for idx in indices:
        _, second = get_key(idx)
        count[second + 1] += 1

    for i in range(1, len(count)):
        count[i] += count[i - 1]

    temp = [0] * n
    # Process in reverse to maintain stability
    for idx in reversed(indices):
        _, second = get_key(idx)
        count[second + 1] -= 1
        temp[count[second + 1]] = idx

    logger.trace(f"  After pass 1: {temp[: min(10, n)]}")

    logger.trace("  Pass 2: sorting by first component")
    count = [0] * (max_value + 1)

    for idx in temp:
        first, _ = get_key(idx)
        count[first] += 1

    for i in range(1, len(count)):
        count[i] += count[i - 1]

    result = [0] * n
    # Process in reverse to maintain stability
    for idx in reversed(temp):
        first, _ = get_key(idx)
        count[first] -= 1
        result[count[first]] = idx

    logger.trace(f"  After pass 2: {result[: min(10, n)]}")
    return result


def suffix_array_doubling(text: str) -> Sequence[int]:
    """
    Construct a suffix array using the doubling algorithm with radix sort.
    """
    logger.info(f"Building suffix array for text: {text!r} (length={len(text)})")

    n = len(text)
    if n == 0:
        logger.debug("Empty text, returning empty suffix array")
        return tuple()

    rank = [ord(c) for c in text]
    logger.debug(f"Initial ranks (by character): {rank}")

    suffix_array = list(range(n))
    logger.debug(f"Initial suffix array: {suffix_array}")

    temp_rank = [0] * n
    k = 1
    iteration = 0

    while k < n:
        iteration += 1
        logger.info(f"=== Iteration {iteration}: comparing {k}-character prefixes ===")

        def sort_key(i: int) -> tuple[int, int]:
            first_half = rank[i]
            second_half = rank[i + k] if i + k < n else -1
            return (first_half, second_half)

        logger.debug(f"Sorting by ({k}, {k})-character pairs using radix sort")
        max_rank = max(rank)
        suffix_array = radix_sort_pairs(suffix_array, sort_key, max_rank)

        for idx, sa_idx in enumerate(suffix_array):
            key = sort_key(sa_idx)
            suffix_preview = text[sa_idx : min(sa_idx + 2 * k, n)]
            logger.trace(
                f"  suffix_array[{idx}] = {sa_idx}: "
                f"key={key}, suffix={suffix_preview!r}"
            )

        # Suffixes with identical (rank[i], rank[i+k]) pairs get same rank
        temp_rank[suffix_array[0]] = 0
        logger.debug(f"Assigning new ranks for {2 * k}-character prefixes")

        for i in range(1, n):
            prev_suffix = suffix_array[i - 1]
            curr_suffix = suffix_array[i]

            if sort_key(curr_suffix) == sort_key(prev_suffix):
                temp_rank[curr_suffix] = temp_rank[prev_suffix]
                logger.trace(
                    f"  suffix[{curr_suffix}] = suffix[{prev_suffix}]: "
                    f"rank={temp_rank[curr_suffix]}"
                )
            else:
                temp_rank[curr_suffix] = temp_rank[prev_suffix] + 1
                logger.trace(
                    f"  suffix[{curr_suffix}] > suffix[{prev_suffix}]: "
                    f"rank={temp_rank[curr_suffix]}"
                )

        rank = temp_rank[:]
        logger.debug(f"Updated ranks: {rank}")

        # Check if all suffixes have unique ranks (early termination)
        if temp_rank[suffix_array[-1]] == n - 1:
            logger.info(
                f"All suffixes have unique ranks after iteration {iteration}, "
                "terminating early"
            )
            break

        k *= 2
        logger.debug(f"Doubling comparison length to {k}")

    logger.info(f"Final suffix array: {suffix_array}")
    logger.debug("Suffix order:")
    for idx, sa_idx in enumerate(suffix_array):
        suffix = text[sa_idx:]
        logger.debug(f"  [{idx}] position {sa_idx}: {suffix!r}")

    return tuple(suffix_array)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("banana", (5, 3, 1, 0, 4, 2)),  # $, a, ana, anana, banana, na, nana
        ("", ()),
        ("a", (0,)),
        ("aa", (1, 0)),  # a, aa
        ("abracadabra", (10, 7, 0, 3, 5, 8, 1, 4, 6, 9, 2)),
        ("mississippi", (10, 7, 4, 1, 0, 9, 8, 6, 3, 5, 2)),
    ],
)
def test_suffix_array_doubling(text: str, expected: Sequence[int]) -> None:
    result = suffix_array_doubling(text)
    assert result == expected, f"Expected {expected}, got {result}"

    if len(result) > 1:
        for i in range(len(result) - 1):
            suffix1 = text[result[i] :]
            suffix2 = text[result[i + 1] :]
            assert suffix1 <= suffix2, (
                f"Suffixes not in order: {suffix1!r} > {suffix2!r}"
            )


cli = click.Group()


@cli.command()
@click.argument("text")
@click.option(
    "--verbose", "-v", count=True, help="Increase verbosity (use -vv for trace)"
)
def build(text: str, verbose: int) -> None:
    """Build and display suffix array for TEXT."""
    logger.remove()
    if verbose >= 2:
        logger.add(lambda msg: click.echo(msg, err=True), level="TRACE")
    elif verbose >= 1:
        logger.add(lambda msg: click.echo(msg, err=True), level="DEBUG")
    else:
        logger.add(lambda msg: click.echo(msg, err=True), level="INFO")

    suffix_array = suffix_array_doubling(text)

    click.echo("\nSuffix Array:")
    click.echo(" ".join(str(i) for i in suffix_array))

    click.echo("\nSuffixes in order:")
    for idx, pos in enumerate(suffix_array):
        click.echo(f"  [{idx}] position {pos}: {text[pos:]!r}")


@cli.command("test")
def run_tests() -> None:
    """Run tests."""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    cli()
