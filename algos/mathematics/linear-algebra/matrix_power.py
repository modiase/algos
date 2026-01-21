#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.numpy -p python313Packages.matplotlib -p python313Packages.pandas -p python313Packages.seaborn -p python313Packages.click -p python313Packages.pytest
from __future__ import annotations

import os
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
import seaborn as sns

type Matrix = npt.NDArray[np.floating]


def squaring_operations(n: int) -> int:
    """Count matrix multiplications for exponentiation by squaring.

    For n in binary, we need:
    - (bit_length - 1) squarings
    - popcount multiplications (one per 1-bit)
    """
    if n <= 1:
        return 0
    return (n.bit_length() - 1) + bin(n).count("1")


def matrix_power_naive(m: Matrix, n: int) -> Matrix:
    """Compute m^n using naive repeated multiplication. O(n) multiplications."""
    if n < 0:
        raise ValueError("Negative exponents not supported")
    if n == 0:
        return np.eye(m.shape[0], dtype=m.dtype)

    result = m.copy()
    for _ in range(n - 1):
        result = result @ m
    return result


def matrix_power_squaring(m: Matrix, n: int) -> Matrix:
    """Compute m^n using exponentiation by squaring. O(log n) multiplications."""
    if n < 0:
        raise ValueError("Negative exponents not supported")
    if n == 0:
        return np.eye(m.shape[0], dtype=m.dtype)

    result = np.eye(m.shape[0], dtype=m.dtype)
    base = m.copy()

    while n > 0:
        if n & 1:
            result = result @ base
        base = base @ base
        n >>= 1

    return result


@pytest.mark.parametrize(
    "matrix_data, power, expected",
    [
        ([[1, 0], [0, 1]], 5, [[1, 0], [0, 1]]),
        ([[2, 3], [4, 5]], 0, [[1, 0], [0, 1]]),
        ([[2, 3], [4, 5]], 1, [[2, 3], [4, 5]]),
        ([[1, 1], [1, 0]], 2, [[2, 1], [1, 1]]),
        ([[1, 1], [1, 0]], 10, [[89, 55], [55, 34]]),
        ([[2, 0], [0, 3]], 4, [[16, 0], [0, 81]]),
    ],
)
def test_matrix_power_naive(
    matrix_data: list[list[int]], power: int, expected: list[list[int]]
) -> None:
    m = np.array(matrix_data, dtype=float)
    result = matrix_power_naive(m, power)
    np.testing.assert_array_almost_equal(result, expected)


@pytest.mark.parametrize(
    "matrix_data, power, expected",
    [
        ([[1, 0], [0, 1]], 5, [[1, 0], [0, 1]]),
        ([[2, 3], [4, 5]], 0, [[1, 0], [0, 1]]),
        ([[2, 3], [4, 5]], 1, [[2, 3], [4, 5]]),
        ([[1, 1], [1, 0]], 2, [[2, 1], [1, 1]]),
        ([[1, 1], [1, 0]], 10, [[89, 55], [55, 34]]),
        ([[2, 0], [0, 3]], 4, [[16, 0], [0, 81]]),
    ],
)
def test_matrix_power_squaring(
    matrix_data: list[list[int]], power: int, expected: list[list[int]]
) -> None:
    m = np.array(matrix_data, dtype=float)
    result = matrix_power_squaring(m, power)
    np.testing.assert_array_almost_equal(result, expected)


def test_both_methods_agree() -> None:
    rng = np.random.default_rng(42)
    for _ in range(10):
        m = rng.random((4, 4))
        power = int(rng.integers(1, 50))
        naive_result = matrix_power_naive(m, power)
        squaring_result = matrix_power_squaring(m, power)
        np.testing.assert_allclose(naive_result, squaring_result, rtol=1e-10)


def run_benchmark(
    matrix_size: int,
    powers: Sequence[int],
    trials: int,
) -> pd.DataFrame:
    results = []
    rng = np.random.default_rng(42)
    m = rng.random((matrix_size, matrix_size))

    for power in powers:
        for _ in range(trials):
            start = time.perf_counter()
            _ = matrix_power_naive(m, power)
            naive_time = time.perf_counter() - start
            results.append(
                {"Power": power, "Algorithm": "Naive", "Time (s)": naive_time}
            )

            start = time.perf_counter()
            _ = matrix_power_squaring(m, power)
            squaring_time = time.perf_counter() - start
            results.append(
                {"Power": power, "Algorithm": "Squaring", "Time (s)": squaring_time}
            )

    return pd.DataFrame(results)


def plot_benchmark(df: pd.DataFrame, output_path: Path, matrix_size: int) -> None:
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    sns.lineplot(
        data=df,
        x="Power",
        y="Time (s)",
        hue="Algorithm",
        marker="o",
        ax=ax,
        errorbar="sd",
    )

    ax.set_yscale("log")
    ax.set_title(f"Matrix Power ({matrix_size}x{matrix_size}): Naive vs Squaring")
    ax.set_xlabel("Power (n)")
    ax.set_ylabel("Time (seconds)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


cli = click.Group()


@cli.command()
@click.option("--size", "-s", default=4, help="Matrix size (NxN)")
@click.option("--power", "-p", default=100, help="Power to compute")
def demo(size: int, power: int) -> None:
    """Demonstrate matrix power computation with both algorithms."""
    rng = np.random.default_rng(42)
    m = rng.integers(0, 10, size=(size, size)).astype(float)

    click.echo(f"Matrix ({size}x{size}):")
    click.echo(m)
    click.echo(f"\nComputing M^{power}...")

    start = time.perf_counter()
    _ = matrix_power_naive(m, power)
    naive_time = time.perf_counter() - start

    start = time.perf_counter()
    squaring_result = matrix_power_squaring(m, power)
    squaring_time = time.perf_counter() - start

    sq_ops = squaring_operations(power)
    click.echo(f"\nNaive method: {power - 1} multiplications, {naive_time:.6f}s")
    click.echo(f"Squaring method: {sq_ops} multiplications, {squaring_time:.6f}s")
    click.echo(
        f"  {power} = {bin(power)} ({power.bit_length()} bits, {bin(power).count('1')} ones)"
    )
    click.echo(
        f"  = {power.bit_length() - 1} squarings + {bin(power).count('1')} multiplications"
    )
    click.echo(f"Speedup: {naive_time / squaring_time:.2f}x")

    click.echo("\nResult (first 4x4 block):")
    click.echo(squaring_result[:4, :4])


@cli.command()
@click.option("--size", "-s", default=32, help="Matrix size (NxN)")
@click.option("--max-power", "-m", default=1000, help="Maximum power to benchmark")
@click.option("--trials", "-t", default=5, help="Number of trials per configuration")
@click.option("--output", "-o", default=None, help="Output path for plot")
@click.option("--open", "open_file", is_flag=True, help="Open plot in browser")
def benchmark(
    size: int,
    max_power: int,
    trials: int,
    output: str | None,
    open_file: bool,
) -> None:
    """Run performance benchmark comparing naive vs squaring approaches."""
    output_path = (
        Path(output)
        if output
        else Path(tempfile.gettempdir()) / "matrix_power_benchmark.png"
    )

    powers = [p for p in [10, 25, 50, 100, 200, 300, 500, 750, 1000] if p <= max_power]

    click.echo("Running benchmark...")
    click.echo(f"  Matrix size: {size}x{size}")
    click.echo(f"  Powers: {powers}")
    click.echo(f"  Trials: {trials}")

    df = run_benchmark(size, powers, trials)

    click.echo("\nResults (mean time in seconds):")
    summary = df.groupby(["Power", "Algorithm"])["Time (s)"].mean()
    click.echo(summary.to_string())

    plot_benchmark(df, output_path, size)
    click.echo(f"\nPlot saved to: {output_path}")

    if open_file:
        os.system(f"open {output_path}")


@cli.command("test")
def run_tests() -> None:
    """Run pytest on this module."""
    pytest.main([__file__, "-v", "-p", "no:cacheprovider"])


if __name__ == "__main__":
    cli()
