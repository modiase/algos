#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.pytest -p python313Packages.click -p python313Packages.jax -p python313Packages.jaxlib
from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Final

import click
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

EPSILON: Final = 1e-10  # Tolerance for floating-point comparisons


@jax.jit
def horners_rule(coefficients: jax.Array, x: float) -> jax.Array:
    n = coefficients.shape[0]
    result = coefficients[n - 1]

    for i in range(n - 2, -1, -1):
        result = result * x + coefficients[i]

    return result


def format_polynomial(coefficients: Sequence[float]) -> str:
    terms = []
    for i, coeff in enumerate(coefficients):
        if abs(coeff) < EPSILON:
            continue

        if i == 0:
            terms.append(f"{coeff:.3g}")
        elif i == 1:
            if abs(coeff - 1.0) < EPSILON:
                terms.append("x")
            elif abs(coeff + 1.0) < EPSILON:
                terms.append("-x")
            else:
                terms.append(f"{coeff:.3g}x")
        else:
            if abs(coeff - 1.0) < EPSILON:
                terms.append(f"x^{i}")
            elif abs(coeff + 1.0) < EPSILON:
                terms.append(f"-x^{i}")
            else:
                terms.append(f"{coeff:.3g}x^{i}")

    if not terms:
        return "0"

    result = terms[0]
    for term in terms[1:]:
        if term.startswith("-"):
            result += f" - {term[1:]}"
        else:
            result += f" + {term}"

    return result


@pytest.mark.parametrize(
    "coefficients, x, expected",
    [
        ([1, 2, 3], 0, 1),
        ([1, 2, 3], 1, 6),
        ([1, 2, 3], 2, 17),
        ([5, 0, 10], 3, 95),
        ([1, -1, 1, -1], 2, -5),
        ([2], 100, 2),
    ],
)
def test_horners_rule(coefficients: Sequence[float], x: float, expected: float) -> None:
    result = horners_rule(jnp.array(coefficients, dtype=jnp.float64), x)
    assert abs(float(result) - expected) < EPSILON


@pytest.mark.parametrize(
    "coefficients, expected",
    [
        ([1, 2, 3], "1 + 2x + 3x^2"),
        ([5, 0, 10], "5 + 10x^2"),
        ([1, -1, 1, -1], "1 - x + x^2 - x^3"),
        ([0, 0, 1], "x^2"),
        ([2], "2"),
        ([1, 1], "1 + x"),
    ],
)
def test_format_polynomial(coefficients: Sequence[float], expected: str) -> None:
    assert format_polynomial(coefficients) == expected


cli = click.Group()


@cli.command()
@click.option("--seed", default=42, help="Random seed for coefficient generation")
@click.option("--degree", "-d", default=None, type=int, help="Degree of the polynomial")
@click.option(
    "--coefficients",
    "-c",
    default=None,
    type=str,
    help="Comma-separated coefficients (e.g., '1,2,3' for 1 + 2x + 3x^2)",
)
@click.option("--x", default=0.0, help="Point at which to evaluate the polynomial")
def demo(seed: int, degree: int | None, coefficients: str | None, x: float) -> None:
    """Demonstrate Horner's rule for polynomial evaluation."""
    if degree is not None and coefficients is not None:
        raise click.UsageError("Cannot specify both --degree and --coefficients")

    if coefficients is not None:
        coeff_list = [float(c.strip()) for c in coefficients.split(",")]
    else:
        if degree is None:
            degree = 3
        random.seed(seed)
        coeff_list = [random.uniform(-10, 10) for _ in range(degree + 1)]

    polynomial_str = format_polynomial(coeff_list)
    click.echo(f"Polynomial: P(x) = {polynomial_str}")

    jax_coefficients = jnp.array(coeff_list, dtype=jnp.float64)
    result = horners_rule(jax_coefficients, x)

    click.echo(f"P({x}) = {float(result)}")


@cli.command()
def test() -> None:
    """Run the test suite."""
    pytest.main([__file__])


if __name__ == "__main__":
    cli()
