#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.pytest -p python313Packages.click -p python313Packages.jax -p python313Packages.jaxlib
from __future__ import annotations

from typing import Final

import click
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

EPSILON: Final = 1e-10


def compute_roots_of_unity(n: int) -> jax.Array:
    """
    Compute the nth roots of unity: ω⁰, ω¹, ..., ωⁿ⁻¹ where ωₖ = e^(2πik/n).
    These are n evenly spaced points on the unit circle in the complex plane.
    """
    return jnp.exp(2j * jnp.pi * jnp.arange(n) / n)


def dft(coefficients: jax.Array, degree_bound: int) -> jax.Array:
    """
    Evaluate polynomial at the nth roots of unity, converting from coefficient
    representation to point-value representation for FFT-based multiplication.
    Time complexity: O(n²) for naive DFT (O(n log n) for FFT variant).
    """
    padded_coeffs = jnp.pad(
        coefficients,
        (0, max(0, degree_bound - len(coefficients))),
        constant_values=0,
    )
    roots = compute_roots_of_unity(degree_bound)
    results = jnp.zeros(degree_bound, dtype=jnp.complex128)
    for i, omega in enumerate(roots):
        result = padded_coeffs[-1]
        for j in range(len(padded_coeffs) - 2, -1, -1):
            result = result * omega + padded_coeffs[j]
        results = results.at[i].set(result)

    return results


def format_complex(z: complex, precision: int = 6) -> str:
    """Format a complex number as a readable string."""
    real = z.real
    imag = z.imag

    if abs(imag) < EPSILON:
        return f"{real:.{precision}f}"

    if abs(real) < EPSILON:
        if abs(imag - 1.0) < EPSILON:
            return "i"
        if abs(imag + 1.0) < EPSILON:
            return "-i"
        return f"{imag:.{precision}f}i"

    sign = "+" if imag >= 0 else "-"
    if abs(abs(imag) - 1.0) < EPSILON:
        imag_str = "i"
    else:
        imag_str = f"{abs(imag):.{precision}f}i"

    return f"{real:.{precision}f} {sign} {imag_str}"


@pytest.mark.parametrize(
    "n, expected_index, expected_value",
    [
        (1, 0, 1 + 0j),
        (2, 0, 1 + 0j),
        (2, 1, -1 + 0j),
        (4, 0, 1 + 0j),
        (4, 1, 0 + 1j),
        (4, 2, -1 + 0j),
        (4, 3, 0 - 1j),
    ],
)
def test_roots_of_unity(n: int, expected_index: int, expected_value: complex) -> None:
    assert abs(compute_roots_of_unity(n)[expected_index] - expected_value) < EPSILON


def test_dft_constant_polynomial() -> None:
    """DFT of constant polynomial should be constant at all roots."""
    coefficients = jnp.array([5.0], dtype=jnp.float64)
    result = dft(coefficients, 4)

    for val in result:
        assert abs(val - 5.0) < EPSILON


def test_dft_linear_polynomial() -> None:
    """Test DFT of 1 + x at 2nd roots of unity."""
    result = dft(jnp.array([1.0, 1.0], dtype=jnp.float64), 2)
    assert abs(result[0] - 2.0) < EPSILON
    assert abs(result[1] - 0.0) < EPSILON


def test_dft_quadratic_polynomial() -> None:
    """Test DFT of 1 + x + x² at 3rd roots of unity."""
    result = dft(jnp.array([1.0, 1.0, 1.0], dtype=jnp.float64), 3)
    assert abs(result[0] - 3.0) < EPSILON

    omega1 = jnp.exp(2j * jnp.pi / 3)
    assert abs(result[1] - (1 + omega1 + omega1**2)) < EPSILON

    omega2 = jnp.exp(4j * jnp.pi / 3)
    assert abs(result[2] - (1 + omega2 + omega2**2)) < EPSILON


def test_format_complex() -> None:
    """Test complex number formatting."""
    assert format_complex(3.0 + 0j, 2) == "3.00"
    assert format_complex(0.0 + 2j, 2) == "2.00i"
    assert format_complex(0.0 + 1j, 2) == "i"
    assert format_complex(0.0 - 1j, 2) == "-i"
    assert format_complex(3.0 + 4j, 2) == "3.00 + 4.00i"
    assert format_complex(3.0 - 4j, 2) == "3.00 - 4.00i"
    assert format_complex(3.0 + 1j, 2) == "3.00 + i"


cli = click.Group()


@cli.command()
@click.option(
    "--poly",
    "-p",
    default="1,1,1",
    help="Comma-separated polynomial coefficients (e.g., '1,1,1' for 1 + x + x²)",
)
@click.option(
    "--degree-bound",
    "-n",
    default=3,
    help="Degree bound (number of roots of unity to evaluate at)",
)
def demo(poly: str, degree_bound: int) -> None:
    """Demonstrate DFT for polynomial evaluation at roots of unity."""
    coefficients = jnp.array(
        [float(c.strip()) for c in poly.split(",")], dtype=jnp.float64
    )

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

    poly_str = terms[0] if terms else "0"
    for term in terms[1:]:
        if term.startswith("-"):
            poly_str += f" - {term[1:]}"
        else:
            poly_str += f" + {term}"

    click.echo(f"Polynomial: P(x) = {poly_str}")
    click.echo(f"Degree bound: n = {degree_bound}")
    click.echo()

    result = dft(coefficients, degree_bound)
    click.echo(f"DFT evaluation at {degree_bound}th roots of unity:")
    click.echo()

    for k, (omega, value) in enumerate(
        zip(compute_roots_of_unity(degree_bound), result)
    ):
        click.echo(f"  P(ω^{k}) where ω^{k} = {format_complex(complex(omega), 6)}")
        click.echo(f"    = {format_complex(complex(value), 6)}")
        click.echo()


@cli.command()
def test() -> None:
    """Run the test suite."""
    pytest.main([__file__])


if __name__ == "__main__":
    cli()
