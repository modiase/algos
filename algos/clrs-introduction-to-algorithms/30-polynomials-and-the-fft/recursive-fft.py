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


def next_power_of_2(n: int) -> int:
    """Find the smallest power of 2 that is >= n."""
    power = 1
    while power < n:
        power *= 2
    return power


def recursive_fft(coefficients: jax.Array) -> jax.Array:
    """
    Compute DFT using the recursive FFT algorithm in O(n log n) time.
    Input size must be a power of 2.
    """
    n = coefficients.shape[0]

    if n == 1:
        return coefficients

    even = recursive_fft(coefficients[0::2])
    odd = recursive_fft(coefficients[1::2])

    omega = jnp.exp(2j * jnp.pi * jnp.arange(n // 2) / n)
    t = omega * odd

    return jnp.concatenate([even + t, even - t])


def fft(coefficients: jax.Array, degree_bound: int | None = None) -> jax.Array:
    """
    Compute DFT using FFT, automatically padding to next power of 2.
    Time complexity: O(n log n).
    """
    if degree_bound is None:
        degree_bound = len(coefficients)

    n = next_power_of_2(max(degree_bound, len(coefficients)))
    padded = jnp.pad(coefficients, (0, n - len(coefficients)), constant_values=0)

    return recursive_fft(padded)


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


def format_compact(z: complex) -> str:
    """Format a complex number compactly with 1 decimal place."""
    real = z.real
    imag = z.imag

    if abs(imag) < EPSILON:
        return f"{real:.1f}"

    if abs(real) < EPSILON:
        return f"{imag:.1f}j"

    sign = "+" if imag >= 0 else ""
    return f"{real:.1f}{sign}{imag:.1f}j"


@pytest.mark.parametrize(
    "n, expected",
    [
        (1, 1),
        (2, 2),
        (3, 4),
        (4, 4),
        (5, 8),
        (8, 8),
        (9, 16),
        (15, 16),
        (16, 16),
    ],
)
def test_next_power_of_2(n: int, expected: int) -> None:
    assert next_power_of_2(n) == expected


def test_fft_constant_polynomial() -> None:
    """FFT of constant polynomial should be constant at all roots."""
    result = fft(jnp.array([5.0], dtype=jnp.float64), 4)
    for val in result:
        assert abs(val - 5.0) < EPSILON


def test_fft_linear_polynomial() -> None:
    """Test FFT of 1 + x at 2nd roots of unity."""
    result = fft(jnp.array([1.0, 1.0], dtype=jnp.float64), 2)
    assert abs(result[0] - 2.0) < EPSILON
    assert abs(result[1] - 0.0) < EPSILON


def test_fft_quadratic_polynomial() -> None:
    """Test FFT of 1 + x + x² at 4th roots of unity (padded)."""
    result = fft(jnp.array([1.0, 1.0, 1.0], dtype=jnp.float64), 4)
    assert abs(result[0] - 3.0) < EPSILON

    omega1 = jnp.exp(2j * jnp.pi / 4)
    assert abs(result[1] - (1 + omega1 + omega1**2)) < EPSILON

    omega2 = jnp.exp(2j * jnp.pi * 2 / 4)
    assert abs(result[2] - (1 + omega2 + omega2**2)) < EPSILON

    omega3 = jnp.exp(2j * jnp.pi * 3 / 4)
    assert abs(result[3] - (1 + omega3 + omega3**2)) < EPSILON


def test_fft_power_of_2_size() -> None:
    """Test FFT with exact power of 2 size."""
    coefficients = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float64)
    result = fft(coefficients)

    assert result.shape[0] == 4
    assert abs(result[0] - 10.0) < EPSILON


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
    default=None,
    type=int,
    help="Degree bound (will pad to next power of 2)",
)
def demo(poly: str, degree_bound: int | None) -> None:
    """Demonstrate FFT for polynomial evaluation at roots of unity."""
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

    click.echo(f"P(x) = {poly_str}")

    if degree_bound is None:
        degree_bound = len(coefficients)

    n = next_power_of_2(max(degree_bound, len(coefficients)))
    click.echo(f"n = {n}")
    click.echo()

    result = fft(coefficients, degree_bound)

    omega_n = jnp.exp(2j * jnp.pi / n)
    for k in range(n):
        omega_k = omega_n**k
        click.echo(
            f"P(ω^{k}) = {format_complex(complex(result[k]), 6)} "
            f"where ω^{k} = {format_complex(complex(omega_k), 6)}"
        )

    click.echo()
    formatted = ", ".join(format_compact(complex(v)) for v in result)
    click.echo(f"y = ({formatted})")


@cli.command()
def test() -> None:
    """Run the test suite."""
    pytest.main([__file__])


if __name__ == "__main__":
    cli()
