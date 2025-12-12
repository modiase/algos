#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.pytest -p python313Packages.click
from __future__ import annotations

import click
import pytest

from extended_euclid import extended_euclid


def modular_linear_equation_solver(a: int, b: int, n: int) -> tuple[int, ...]:
    """
    Solve modular linear equation ax ≡ b (mod n).

    Returns all solutions in ascending order, or empty tuple if gcd(a, n) ∤ b.
    When gcd(a, n) | b, there are exactly gcd(a, n) solutions.

    Time: O(log n), Space: O(gcd(a, n))
    """
    d, x0, _ = extended_euclid(a, n)

    if b % d != 0:
        return ()

    x0 = (x0 * (b // d)) % n
    return tuple(sorted((x0 + i * (n // d)) % n for i in range(d)))


@pytest.mark.parametrize(
    "a, b, n, expected",
    [
        # Single solution cases (gcd(a, n) = 1)
        (3, 5, 7, (4,)),  # 3*4 = 12 ≡ 5 (mod 7)
        (5, 3, 7, (2,)),  # 5*2 = 10 ≡ 3 (mod 7)
        (7, 1, 10, (3,)),  # 7*3 = 21 ≡ 1 (mod 10)
        # Multiple solution cases (gcd(a, n) > 1)
        (14, 30, 100, (45, 95)),  # gcd(14, 100) = 2
        (6, 9, 15, (4, 9, 14)),  # gcd(6, 15) = 3
        (10, 5, 25, (3, 8, 13, 18, 23)),  # gcd(10, 25) = 5
        (4, 2, 6, (2, 5)),  # gcd(4, 6) = 2
        # No solution cases (gcd doesn't divide b)
        (2, 3, 4, ()),  # gcd(2, 4) = 2 doesn't divide 3
        (6, 4, 9, ()),  # gcd(6, 9) = 3 doesn't divide 4
        (4, 3, 8, ()),  # gcd(4, 8) = 4 doesn't divide 3
        # Edge cases
        (1, 5, 7, (5,)),  # a = 1 always has solution x = b
        (0, 0, 5, (0, 1, 2, 3, 4)),  # 0x ≡ 0 (mod 5) - all x are solutions
        (5, 0, 10, (0, 2, 4, 6, 8)),  # gcd(5, 10) = 5
    ],
)
def test_modular_linear_equation_solver(
    a: int, b: int, n: int, expected: tuple[int, ...]
) -> None:
    """Test the modular linear equation solver."""
    solutions = modular_linear_equation_solver(a, b, n)
    assert solutions == expected

    # Verify each solution satisfies the equation
    for x in solutions:
        assert (a * x) % n == b % n, (
            f"Solution {x} doesn't satisfy {a}*{x} ≡ {b} (mod {n})"
        )


def test_modular_linear_equation_solver_comprehensive() -> None:
    """Additional verification tests."""
    # Test that all solutions are unique and in range [0, n)
    solutions = modular_linear_equation_solver(14, 30, 100)
    assert len(solutions) == len(set(solutions)), "Solutions should be unique"
    assert all(0 <= x < 100 for x in solutions), "Solutions should be in [0, n)"

    # Test prime modulus (should always have single solution when a != 0)
    for a in range(1, 11):
        for b in range(11):
            assert len(modular_linear_equation_solver(a, b, 11)) == 1, (
                f"Prime modulus should give single solution for a={a}, b={b}"
            )


@click.group()
def cli() -> None:
    """Solve modular linear equations ax ≡ b (mod n)."""
    pass


@cli.command()
@click.option("-a", "--a-coeff", type=int, default=14, help="Coefficient a")
@click.option("-b", "--b-value", type=int, default=30, help="Value b")
@click.option("-n", "--modulus", type=int, default=100, help="Modulus n")
def demo(a_coeff: int, b_value: int, modulus: int) -> None:
    """Demonstrate solving a modular linear equation."""
    click.echo(f"Solving: {a_coeff}x ≡ {b_value} (mod {modulus})")
    click.echo()

    solutions = modular_linear_equation_solver(a_coeff, b_value, modulus)

    if solutions:
        click.echo(f"Found {len(solutions)} solution(s):")
        for x in solutions:
            click.echo(
                f"  x = {x:3d}  (verification: {a_coeff}*{x} ≡ {(a_coeff * x) % modulus} (mod {modulus}))"
            )
    else:
        click.echo(
            f"No solutions (gcd({a_coeff}, {modulus}) = {extended_euclid(a_coeff, modulus)[0]} does not divide {b_value})"
        )


@cli.command()
def test() -> None:
    """Run the test suite."""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    cli()
