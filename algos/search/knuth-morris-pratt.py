#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.loguru -p python313Packages.pytest -p python313Packages.click
from __future__ import annotations

from collections.abc import Sequence

import click
import pytest
from loguru import logger


def compute_lps(pattern: str) -> Sequence[int]:
    """Compute longest proper prefix which is also suffix array."""
    m = len(pattern)
    lps = [0] * m
    length = 0
    i = 1

    logger.debug(f"Computing LPS for pattern: {pattern!r}")

    while i < m:
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            logger.trace(f"  lps[{i}] = {length} (match at {pattern[i]!r})")
            i += 1
        elif length != 0:
            length = lps[length - 1]
            logger.trace(f"  fallback to length={length}")
        else:
            lps[i] = 0
            logger.trace(f"  lps[{i}] = 0")
            i += 1

    logger.debug(f"LPS array: {lps}")
    return tuple(lps)


def kmp_search(text: str, pattern: str) -> Sequence[int]:
    """Find all occurrences of pattern in text using KMP algorithm."""
    logger.info(f"Searching for {pattern!r} in text of length {len(text)}")

    if pattern == "":
        return tuple(range(len(text) + 1))

    n = len(text)
    m = len(pattern)

    if m > n:
        return tuple()

    lps = compute_lps(pattern)
    matches: list[int] = []

    i = 0
    j = 0
    while i < n:
        if pattern[j] == text[i]:
            i += 1
            j += 1
            logger.trace(f"  match at text[{i - 1}], j={j}")

        if j == m:
            logger.debug(f"Found match at position {i - j}")
            matches.append(i - j)
            j = lps[j - 1]
        elif i < n and pattern[j] != text[i]:
            if j != 0:
                logger.trace(f"  mismatch, fallback j={lps[j - 1]}")
                j = lps[j - 1]
            else:
                logger.trace("  mismatch at start, advance i")
                i += 1

    logger.info(f"Found {len(matches)} matches")
    return tuple(matches)


@pytest.mark.parametrize(
    ("pattern", "expected_lps"),
    [
        ("AAAA", (0, 1, 2, 3)),
        ("ABCDE", (0, 0, 0, 0, 0)),
        ("AABAACAABAA", (0, 1, 0, 1, 2, 0, 1, 2, 3, 4, 5)),
        ("AAACAAAAAC", (0, 1, 2, 0, 1, 2, 3, 3, 3, 4)),
        ("AABAAAB", (0, 1, 0, 1, 2, 2, 3)),
    ],
)
def test_compute_lps(pattern: str, expected_lps: Sequence[int]) -> None:
    assert compute_lps(pattern) == expected_lps


@pytest.mark.parametrize(
    ("text", "pattern", "expected"),
    [
        ("abracadabra", "abra", (0, 7)),
        ("aaaaa", "aa", (0, 1, 2, 3)),
        ("abcdef", "gh", ()),
        ("", "", (0,)),
        ("a" * 100 + "b", "ab", (99,)),
        ("ABABDABACDABABCABAB", "ABABCABAB", (10,)),
        ("AAAAABAAABA", "AAAA", (0, 1)),
    ],
)
def test_kmp_search(text: str, pattern: str, expected: Sequence[int]) -> None:
    assert kmp_search(text, pattern) == expected


cli = click.Group()


@cli.command()
@click.argument("text")
@click.argument("pattern")
@click.option("--verbose", "-v", count=True, help="Increase verbosity")
def search(text: str, pattern: str, verbose: int) -> None:
    """Search for PATTERN in TEXT using KMP algorithm."""
    logger.remove()
    if verbose >= 2:
        logger.add(lambda msg: click.echo(msg, err=True), level="TRACE")
    elif verbose >= 1:
        logger.add(lambda msg: click.echo(msg, err=True), level="DEBUG")
    else:
        logger.add(lambda msg: click.echo(msg, err=True), level="INFO")

    matches = kmp_search(text, pattern)
    click.echo(" ".join(str(i) for i in matches) or "no matches")


@cli.command("test")
def run_tests() -> None:
    """Run tests."""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    cli()
