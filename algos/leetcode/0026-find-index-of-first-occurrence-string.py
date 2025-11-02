#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.pytest -p python313Packages.click -p python313Packages.loguru
from __future__ import annotations

import click
import pytest
from loguru import logger


def strStr(haystack: str, needle: str) -> int:
    """
    The naive implementation is quadratic. This is a linear solution using
    KMP search.
    """
    M, N = len(haystack), len(needle)
    if N == 0:
        return 0
    if M < N:
        return -1

    lps = [0] * N
    j = 0
    for i in range(1, N):
        while j > 0 and needle[i] != needle[j]:
            j = lps[j - 1]

        if needle[i] == needle[j]:
            j += 1

        lps[i] = j

    j = 0
    i = 0
    while i < M:
        logger.debug(haystack)
        logger.debug(" " * (i - j) + needle)
        logger.debug(" " * (i - j) + "-" * j)
        logger.debug(" " * i + "*")
        logger.debug("")

        if haystack[i] == needle[j]:
            j += 1
            i += 1
            if j == N:
                return i - N
        elif j == 0:
            i += 1
        else:
            j = lps[j - 1]

    return -1


@pytest.mark.parametrize(
    "haystack, needle, expected",
    [
        ("sadbutsad", "sad", 0),
        ("leetcode", "leeto", -1),
        ("hello", "ll", 2),
        ("aaaaa", "bba", -1),
        ("", "a", -1),
        ("a", "", 0),
        ("mississippi", "issip", 4),
        ("aabaaabaaac", "aabaaac", 4),
        ("ababcaababcaabc", "ababcaabc", 6),
        ("abcdefgh", "cde", 2),
        ("abcdefgh", "xyz", -1),
        ("a", "a", 0),
        ("abc", "c", 2),
    ],
)
def test_strStr(haystack: str, needle: str, expected: int) -> None:
    result = strStr(haystack, needle)
    assert result == expected


@click.group()
def cli():
    pass


@cli.command()
@click.argument("haystack")
@click.argument("needle")
def find(haystack: str, needle: str):
    """Find the first occurrence of needle in haystack."""
    result = strStr(haystack, needle)
    if result == -1:
        click.echo(f"'{needle}' not found in '{haystack}'")
    else:
        click.echo(f"'{needle}' found at index {result} in '{haystack}'")


@cli.command("test")
def run_tests():
    """Run tests."""
    pytest.main([__file__])


if __name__ == "__main__":
    cli()
