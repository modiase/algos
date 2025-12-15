#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.pytest -p python313Packages.click
from __future__ import annotations

from functools import reduce

import click
import pytest


def modular_exponentiation(a: int, b: int, n: int) -> int:
    """
    Efficiently computes a^b mod n through representation of b in binary form
    and computing the power through combination of a^k where k is a power of
    2 - computed efficiently through squaring.
    """
    return reduce(
        lambda d, b: d * d * (a if b else 1) % n,
        (1 & (b >> k) for k in range(b.bit_length(), -1, -1)),
        1,
    )


@pytest.mark.parametrize(
    "a, b, n, expected",
    [
        (2, 3, 5, 3),
        (5, 100, 3, 1),
        (7, 13, 11, 2),
        (3, 1000, 100, 1),
        (2, 10, 1000, 24),
        (123, 456, 789, 699),
        (2, 0, 5, 1),
        (0, 5, 7, 0),
        (1, 1000000, 13, 1),
        (7, 560, 561, 1),
        (2, 90, 13, 12),
        (10, 18, 19, 1),
        (12345, 6789, 10007, 2621),
        (2, 1000000, 1000000007, 235042059),
    ],
)
def test_modular_exponentiation(a: int, b: int, n: int, expected: int) -> None:
    result = modular_exponentiation(a, b, n)
    assert result == expected
    assert result == pow(a, b, n)


cli = click.Group()


@cli.command()
@click.option("--a", "-a", default=7, help="Base")
@click.option("--b", "-b", default=13, help="Exponent")
@click.option("--n", "-n", default=11, help="Modulus")
def demo(a: int, b: int, n: int) -> None:
    result = modular_exponentiation(a, b, n)
    click.echo(f"Computing {a}^{b} mod {n}")
    click.echo(f"Result: {result}")
    click.echo(f"Verification (using Python's pow): {pow(a, b, n)}")


@cli.command("test")
def run_tests() -> None:
    pytest.main([__file__])


if __name__ == "__main__":
    cli()
