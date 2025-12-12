#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.pytest -p python313Packages.click
from __future__ import annotations

from functools import reduce
from operator import mul

import click
import pytest

from extended_euclid import extended_euclid


def are_pairwise_coprime(moduli: tuple[int, ...]) -> bool:
    """
    Check if all moduli are pairwise coprime: gcd(mᵢ, mⱼ) = 1 for all i ≠ j.

    Time: O(k² log max(moduli)) where k = len(moduli)
    """
    for i in range(len(moduli)):
        for j in range(i + 1, len(moduli)):
            if extended_euclid(moduli[i], moduli[j])[0] != 1:
                return False
    return True


def chinese_remainder_theorem(equations: tuple[tuple[int, int], ...]) -> int | None:
    """
    Solve system of congruences x ≡ aᵢ (mod nᵢ) using CRT.

    Returns unique solution modulo ∏nᵢ, or None if moduli not pairwise coprime.

    Algorithm: x = (Σ aᵢ × Nᵢ × Mᵢ) mod N where N = ∏nᵢ, Nᵢ = N/nᵢ, Nᵢ×Mᵢ ≡ 1 (mod nᵢ).

    Time: O(k² log max(moduli)), Space: O(1)
    """
    if not equations:
        return None

    _, moduli = zip(*equations)

    if not are_pairwise_coprime(moduli):
        return None

    N = reduce(mul, moduli)
    result = 0

    for a_i, n_i in equations:
        N_i = N // n_i
        gcd, M_i, _ = extended_euclid(N_i, n_i)
        assert gcd == 1
        result += a_i * N_i * M_i

    return result % N


@pytest.mark.parametrize(
    "moduli, expected",
    [
        ((3, 5, 7), True),  # All pairwise coprime
        ((2, 3, 5), True),  # All pairwise coprime
        ((4, 9, 25), True),  # Powers of different primes
        ((6, 10), False),  # gcd(6, 10) = 2
        ((12, 18), False),  # gcd(12, 18) = 6
        ((4, 6, 9), False),  # gcd(4, 6) = 2
        ((5,), True),  # Single modulus
        ((11, 13, 17), True),  # All prime
    ],
)
def test_are_pairwise_coprime(moduli: tuple[int, ...], expected: bool) -> None:
    assert are_pairwise_coprime(moduli) == expected


@pytest.mark.parametrize(
    "equations, expected",
    [
        # Classic example: x ≡ 2 (mod 3), x ≡ 3 (mod 5), x ≡ 2 (mod 7)
        (((2, 3), (3, 5), (2, 7)), 23),
        # Two equations: x ≡ 1 (mod 3), x ≡ 4 (mod 5)
        (((1, 3), (4, 5)), 4),
        # x ≡ 0 (mod 3), x ≡ 0 (mod 5), x ≡ 0 (mod 7)
        (((0, 3), (0, 5), (0, 7)), 0),
        # x ≡ 1 (mod 2), x ≡ 2 (mod 3), x ≡ 3 (mod 5)
        (((1, 2), (2, 3), (3, 5)), 23),
        # Single equation: x ≡ 5 (mod 7)
        (((5, 7),), 5),
        # x ≡ 3 (mod 4), x ≡ 5 (mod 9), x ≡ 7 (mod 25)
        (((3, 4), (5, 9), (7, 25)), 707),
    ],
)
def test_chinese_remainder_theorem(
    equations: tuple[tuple[int, int], ...], expected: int
) -> None:
    result = chinese_remainder_theorem(equations)
    assert result == expected

    for a_i, n_i in equations:
        assert result % n_i == a_i % n_i


def test_chinese_remainder_theorem_not_coprime() -> None:
    """Test that non-coprime moduli return None."""
    assert chinese_remainder_theorem(((2, 4), (3, 6))) is None
    assert chinese_remainder_theorem(((1, 6), (2, 10))) is None
    assert chinese_remainder_theorem(((1, 12), (5, 18))) is None


def test_chinese_remainder_theorem_empty() -> None:
    """Test empty input."""
    assert chinese_remainder_theorem(()) is None


@click.group()
def cli() -> None:
    """Solve systems of congruences using Chinese Remainder Theorem."""
    pass


@cli.command()
@click.option(
    "-e",
    "--equation",
    "equations",
    type=(int, int),
    multiple=True,
    help="Equation as (residue, modulus) pair. Can be specified multiple times.",
)
def demo(equations: tuple[tuple[int, int], ...]) -> None:
    """Demonstrate solving a system of congruences using CRT."""
    if not equations:
        equations = ((2, 3), (3, 5), (2, 7))
        click.echo("Using default example:")

    click.echo("System of congruences:")
    for i, (a, n) in enumerate(equations, 1):
        click.echo(f"  ({i}) x ≡ {a} (mod {n})")
    click.echo()

    _, moduli = zip(*equations)

    if not are_pairwise_coprime(moduli):
        click.echo("ERROR: Moduli are not pairwise coprime!")
        click.echo(
            "The Chinese Remainder Theorem requires all moduli to be pairwise coprime."
        )
        click.echo()
        for i in range(len(moduli)):
            for j in range(i + 1, len(moduli)):
                if (gcd := extended_euclid(moduli[i], moduli[j])[0]) != 1:
                    click.echo(f"  gcd({moduli[i]}, {moduli[j]}) = {gcd} ≠ 1")
        return

    result = chinese_remainder_theorem(equations)

    click.echo(f"Solution: x = {result} (mod {reduce(mul, moduli)})")
    click.echo()
    click.echo("Verification:")
    for a, n in equations:
        status = "✓" if (result % n) == (a % n) else "✗"
        click.echo(f"  {status} {result} ≡ {result % n} (mod {n}) [expected {a % n}]")


@cli.command()
def test() -> None:
    """Run the test suite."""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    cli()
