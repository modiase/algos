#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.pytest -p python313Packages.click -p python313Packages.numpy
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Final

import click
import numpy as np
import pytest
from numpy.typing import NDArray

EPSILON: Final = 1e-10
DEFAULT_COARSENING: Final = 1024


def bit_reverse_permutation(n: int) -> NDArray:
    """Compute bit-reversal permutation indices for size n."""
    bits = n.bit_length() - 1
    return np.array([int(bin(i)[2:].zfill(bits)[::-1], 2) for i in range(n)])


def butterfly_stage(
    args: tuple[NDArray, int, int, complex],
) -> list[tuple[int, complex, complex]]:
    """Compute butterfly operations for a single block."""
    a, k, m, omega_m = args
    results = []
    omega = 1.0
    for j in range(m // 2):
        t = omega * a[k + j + m // 2]
        u = a[k + j]
        results.append((k + j, u + t, u - t))
        omega *= omega_m
    return results


def parallel_iterative_fft(
    coefficients: NDArray,
    num_workers: int | None = None,
    coarsening: int = DEFAULT_COARSENING,
) -> NDArray:
    """
    Compute DFT using parallel iterative FFT in O(n log n) time.
    Parallelizes butterfly blocks across multiple processes.
    Input size must be a power of 2.
    """
    n = len(coefficients)
    a = coefficients[bit_reverse_permutation(n)].astype(np.complex128)

    m = 2
    while m <= n:
        omega_m = np.exp(2j * np.pi / m)
        num_blocks = n // m

        if num_blocks >= 2 and m >= coarsening:
            blocks = [(a.copy(), k, m, omega_m) for k in range(0, n, m)]

            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                all_results = list(executor.map(butterfly_stage, blocks))

                for results in all_results:
                    for idx, val_upper, val_lower in results:
                        a[idx] = val_upper
                        a[idx + m // 2] = val_lower
        else:
            for k in range(0, n, m):
                omega = 1.0
                for j in range(m // 2):
                    t = omega * a[k + j + m // 2]
                    u = a[k + j]
                    a[k + j] = u + t
                    a[k + j + m // 2] = u - t
                    omega *= omega_m

        m *= 2

    return a


def fft(
    coefficients: NDArray,
    degree_bound: int | None = None,
    num_workers: int | None = None,
    coarsening: int = DEFAULT_COARSENING,
) -> NDArray:
    """
    Compute DFT using parallel iterative FFT, automatically padding to next power of 2.
    Time complexity: O(n log n).
    """
    if degree_bound is None:
        degree_bound = len(coefficients)

    n = 1
    while n < max(degree_bound, len(coefficients)):
        n *= 2

    padded = np.pad(coefficients, (0, n - len(coefficients)), constant_values=0)
    return parallel_iterative_fft(padded, num_workers, coarsening)


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
    "n",
    [1, 2, 4, 8, 16],
)
def test_bit_reverse_permutation(n: int) -> None:
    perm = bit_reverse_permutation(n)
    assert len(perm) == n
    assert len(set(perm)) == n
    assert all(0 <= i < n for i in perm)


def test_fft_constant_polynomial() -> None:
    """FFT of constant polynomial should be constant at all roots."""
    result = fft(np.array([5.0], dtype=np.float64), 4)
    for val in result:
        assert abs(val - 5.0) < EPSILON


def test_fft_linear_polynomial() -> None:
    """Test FFT of 1 + x at 2nd roots of unity."""
    result = fft(np.array([1.0, 1.0], dtype=np.float64), 2)
    assert abs(result[0] - 2.0) < EPSILON
    assert abs(result[1] - 0.0) < EPSILON


def test_fft_power_of_2_size() -> None:
    """Test FFT with exact power of 2 size."""
    coefficients = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    result = fft(coefficients)

    assert result.shape[0] == 4
    assert abs(result[0] - 10.0) < EPSILON


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
@click.option(
    "--workers",
    "-w",
    default=None,
    type=int,
    help="Number of worker processes (default: CPU count)",
)
@click.option(
    "--coarsening",
    "-c",
    default=DEFAULT_COARSENING,
    type=int,
    help=f"Minimum block size for parallelization (default: {DEFAULT_COARSENING})",
)
def demo(
    poly: str, degree_bound: int | None, workers: int | None, coarsening: int
) -> None:
    """Demonstrate parallel iterative FFT for polynomial evaluation at roots of unity."""
    coefficients = np.array(
        [float(c.strip()) for c in poly.split(",")], dtype=np.float64
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

    n = 1
    while n < max(degree_bound, len(coefficients)):
        n *= 2

    click.echo(f"n = {n}")
    click.echo()

    result = fft(coefficients, degree_bound, workers, coarsening)

    for k in range(n):
        click.echo(f"P(ω^{k}) = {format_compact(complex(result[k]))}")

    click.echo()
    formatted = ", ".join(format_compact(complex(v)) for v in result)
    click.echo(f"y = ({formatted})")


@cli.command()
def test() -> None:
    """Run the test suite."""
    pytest.main([__file__])


if __name__ == "__main__":
    cli()
