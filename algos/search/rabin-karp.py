#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.loguru -p python313Packages.pytest -p python313Packages.pyvis -p python313Packages.more-itertools -p python313Packages.click
from __future__ import annotations

from collections.abc import Sequence

import click
import pytest

BASE = 256
MODULUS = (1 << 31) - 1


def rabin_karp(text: str, pattern: str) -> Sequence[int]:
    if pattern == "":
        return tuple(range(len(text) + 1))
    pattern_length = len(pattern)
    if pattern_length > len(text):
        return tuple()

    pattern_hash = 0
    window_hash = 0
    for index in range(pattern_length):
        pattern_hash = (pattern_hash * BASE + ord(pattern[index])) % MODULUS
        window_hash = (window_hash * BASE + ord(text[index])) % MODULUS

    highest_power = pow(BASE, pattern_length - 1, MODULUS)
    matches: list[int] = []
    limit = len(text) - pattern_length

    for start in range(limit + 1):
        if (
            window_hash == pattern_hash
            and text[start : start + pattern_length] == pattern
        ):
            matches.append(start)
        if start == limit:
            break
        leading = ord(text[start])
        trailing = ord(text[start + pattern_length])
        window_hash = (window_hash - leading * highest_power) % MODULUS
        window_hash = (window_hash * BASE + trailing) % MODULUS

    return tuple(matches)


@pytest.mark.parametrize(
    ("text", "pattern", "expected"),
    [
        ("abracadabra", "abra", (0, 7)),
        ("aaaaa", "aa", (0, 1, 2, 3)),
        ("abcdef", "gh", ()),
        ("", "", (0,)),
        ("a" * 100 + "b", "ab", (99,)),
    ],
)
def test_rabin_karp(text: str, pattern: str, expected: Sequence[int]) -> None:
    assert rabin_karp(text, pattern) == expected


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.argument("text")
@click.argument("pattern")
def search(text: str, pattern: str) -> None:
    click.echo(
        " ".join(str(index) for index in rabin_karp(text, pattern)) or "no matches"
    )


@cli.command("test")
def run_tests() -> None:
    pytest.main([__file__])


if __name__ == "__main__":
    cli()
