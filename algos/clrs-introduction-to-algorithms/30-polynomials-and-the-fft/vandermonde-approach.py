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

EPSILON: Final = 1e-10


def vandermonde_interpolate(x_points: jax.Array, y_points: jax.Array) -> jax.Array:
    """
    Find polynomial coefficients by solving the Vandermonde system V·a = y.

    WARNING: This naive implementation is O(n³) time and O(n²) space with exponentially
    growing condition numbers (≈O(2ⁿ)), causing severe numerical instability for n > 20.

    NOTE: Faster O(n²) algorithms exist:
    - Traub (1966): Inverts Vandermonde matrices; Parker-Traub variant adds stability
    - Björck-Pereyra (1970): Solves systems via Newton interpolation with better accuracy

    For practical use, prefer Lagrange or Newton interpolation to avoid matrix formation.
    """
    n = x_points.shape[0]
    V = jnp.vander(x_points, N=n, increasing=True)
    coefficients = jnp.linalg.solve(V, y_points)

    return coefficients


def evaluate_polynomial(coefficients: jax.Array, x: float) -> jax.Array:
    """Evaluate polynomial at x using Horner's rule."""
    n = coefficients.shape[0]
    result = coefficients[n - 1]

    for i in range(n - 2, -1, -1):
        result = result * x + coefficients[i]

    return result


# Tests
@pytest.mark.parametrize(
    "x_points, y_points, test_x, expected_y",
    [
        # Linear: y = 2x + 1
        ([0.0, 1.0], [1.0, 3.0], 2.0, 5.0),
        # Quadratic: y = x² - 2x + 1 = (x - 1)²
        ([0.0, 1.0, 2.0], [1.0, 0.0, 1.0], 3.0, 4.0),
        # Cubic: y = x³ (passes through origin and cubes)
        ([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 8.0, 27.0], 4.0, 64.0),
        # Quadratic with negative values
        ([-1.0, 0.0, 1.0], [1.0, 0.0, 1.0], 2.0, 4.0),
    ],
)
def test_vandermonde_interpolate(
    x_points: Sequence[float],
    y_points: Sequence[float],
    test_x: float,
    expected_y: float,
) -> None:
    x_arr = jnp.array(x_points, dtype=jnp.float64)
    y_arr = jnp.array(y_points, dtype=jnp.float64)

    coefficients = vandermonde_interpolate(x_arr, y_arr)

    # Test that interpolated polynomial passes through all given points
    for x, y in zip(x_points, y_points):
        result = evaluate_polynomial(coefficients, x)
        assert abs(float(result) - y) < EPSILON, f"Failed at point ({x}, {y})"

    # Test evaluation at test point
    result = evaluate_polynomial(coefficients, test_x)
    assert abs(float(result) - expected_y) < EPSILON


def test_constant_polynomial() -> None:
    """Test interpolation of a constant function."""
    x_points = jnp.array([0.0, 1.0, 2.0], dtype=jnp.float64)
    y_points = jnp.array([5.0, 5.0, 5.0], dtype=jnp.float64)

    coefficients = vandermonde_interpolate(x_points, y_points)

    # Should get coefficients [5, 0, 0] (approximately)
    assert abs(float(coefficients[0]) - 5.0) < EPSILON
    assert abs(float(coefficients[1])) < EPSILON
    assert abs(float(coefficients[2])) < EPSILON


def test_single_point() -> None:
    """Test interpolation through a single point (constant polynomial)."""
    x_points = jnp.array([2.0], dtype=jnp.float64)
    y_points = jnp.array([7.0], dtype=jnp.float64)

    coefficients = vandermonde_interpolate(x_points, y_points)

    # Should get coefficient [7]
    assert abs(float(coefficients[0]) - 7.0) < EPSILON

    # Verify it evaluates correctly at various points
    for x in [0.0, 1.0, 2.0, 10.0]:
        result = evaluate_polynomial(coefficients, x)
        assert abs(float(result) - 7.0) < EPSILON


# CLI
cli = click.Group()


@cli.command()
@click.option("--seed", default=42, help="Random seed for point generation")
@click.option(
    "--points",
    "-p",
    default=None,
    type=str,
    help="Points as 'x1,y1;x2,y2;...' (e.g., '0,1;1,3;2,5')",
)
@click.option("--n", default=4, help="Number of random points to generate")
@click.option("--x", default=None, type=float, help="Point at which to evaluate")
def demo(seed: int, points: str | None, n: int, x: float | None) -> None:
    """Demonstrate Vandermonde polynomial interpolation."""
    if points is not None:
        # Parse points
        point_pairs = [p.strip().split(",") for p in points.split(";")]
        x_points = jnp.array([float(p[0]) for p in point_pairs], dtype=jnp.float64)
        y_points = jnp.array([float(p[1]) for p in point_pairs], dtype=jnp.float64)
    else:
        # Generate random points
        random.seed(seed)
        x_points = jnp.array([float(i) for i in range(n)], dtype=jnp.float64)
        y_points = jnp.array(
            [random.uniform(-10, 10) for _ in range(n)], dtype=jnp.float64
        )

    click.echo(f"Interpolating through {len(x_points)} points:")
    for x_val, y_val in zip(x_points, y_points):
        click.echo(f"  ({float(x_val):.3f}, {float(y_val):.3f})")

    coefficients = vandermonde_interpolate(x_points, y_points)

    click.echo("\nPolynomial coefficients [a₀, a₁, ..., aₙ₋₁]:")
    for i, c in enumerate(coefficients):
        click.echo(f"  a_{i} = {float(c):.6f}")

    # Evaluate at a point
    if x is None:
        x = float(x_points[-1]) + 1.0

    result = evaluate_polynomial(coefficients, x)
    click.echo(f"\nP({x}) = {float(result):.6f}")


@cli.command()
def test() -> None:
    """Run the test suite."""
    pytest.main([__file__])


if __name__ == "__main__":
    cli()
