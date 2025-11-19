#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.pytest -p python313Packages.click -p python313Packages.loguru -p python313Packages.numpy
from __future__ import annotations

import random
from collections.abc import Sequence

import click
import numpy as np
import pytest
from loguru import logger


def log_matrix(matrix: np.ndarray, precision: int = 3) -> str:
    rows = []
    for row in matrix:
        formatted_values = [f"{val:>{precision + 4}.{precision}f}" for val in row]
        rows.append("  [" + "  ".join(formatted_values) + "]")
    return "\n" + "\n".join(rows)


def lu_decomposition(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if a.shape[0] != a.shape[1]:
        raise ValueError("Matrix must be square")

    n = a.shape[0]
    logger.trace(f"Starting LU decomposition for {n}×{n} matrix")

    l_matrix = np.zeros((n, n), dtype=np.float64)
    u_matrix = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        l_matrix[i, i] = 1.0

    a_copy = a.copy()

    for k in range(n):
        logger.trace(f"Processing column {k}")
        u_matrix[k, k] = a_copy[k, k]

        for i in range(k + 1, n):
            l_matrix[i, k] = a_copy[i, k] / u_matrix[k, k]
            u_matrix[k, i] = a_copy[k, i]
        logger.trace(f"L matrix:{log_matrix(l_matrix)}")
        logger.trace(f"U matrix:{log_matrix(u_matrix)}")

        for i in range(k + 1, n):
            for j in range(k + 1, n):
                a_copy[i, j] = a_copy[i, j] - l_matrix[i, k] * u_matrix[k, j]
        logger.trace(f"Updated A:{log_matrix(a_copy)}")

    logger.trace("LU decomposition complete")
    return l_matrix, u_matrix


def lu_solve(l_matrix: np.ndarray, u_matrix: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = l_matrix.shape[0]
    logger.trace(f"Solving LUx = b for {n}×{n} system")

    y = np.zeros(n, dtype=np.float64)
    for i in range(n):
        y[i] = b[i] - sum(l_matrix[i, j] * y[j] for j in range(i))
    logger.trace(f"Forward substitution: y = {y}")

    x = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(u_matrix[i, j] * x[j] for j in range(i + 1, n))) / u_matrix[
            i, i
        ]
    logger.trace(f"Backward substitution: x = {x}")

    return x


def lu_invert(a: np.ndarray) -> np.ndarray:
    n = a.shape[0]
    logger.trace(f"Computing inverse of {n}×{n} matrix")

    l_matrix, u_matrix = lu_decomposition(a)

    inverse = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        e_i = np.zeros(n, dtype=np.float64)
        e_i[i] = 1.0
        logger.trace(f"Solving for column {i}: e_{i}")
        inverse[:, i] = lu_solve(l_matrix, u_matrix, e_i)

    return inverse


@pytest.mark.parametrize(
    "a_data, expected_l, expected_u",
    [
        (
            [[2, 3], [4, 7]],
            [[1, 0], [2, 1]],
            [[2, 3], [0, 1]],
        ),
        (
            [[4, 3], [6, 3]],
            [[1, 0], [1.5, 1]],
            [[4, 3], [0, -1.5]],
        ),
    ],
)
def test_lu_decomposition(
    a_data: Sequence[Sequence[float]],
    expected_l: Sequence[Sequence[float]],
    expected_u: Sequence[Sequence[float]],
) -> None:
    a = np.array(a_data, dtype=np.float64)
    expected_l_array = np.array(expected_l, dtype=np.float64)
    expected_u_array = np.array(expected_u, dtype=np.float64)

    l_matrix, u_matrix = lu_decomposition(a)

    np.testing.assert_array_almost_equal(l_matrix, expected_l_array)
    np.testing.assert_array_almost_equal(u_matrix, expected_u_array)


@pytest.mark.parametrize(
    "a_data",
    [
        [[2, 3], [4, 7]],
        [[1, 2, 3], [2, 5, 2], [3, 1, 5]],
        [[4, 3], [6, 3]],
        [[1, 2, 3, 4], [2, 5, 2, 1], [3, 1, 5, 2], [4, 1, 2, 6]],
    ],
)
def test_lu_reconstruction(a_data: Sequence[Sequence[float]]) -> None:
    a = np.array(a_data, dtype=np.float64)
    l_matrix, u_matrix = lu_decomposition(a)
    reconstructed = l_matrix @ u_matrix
    np.testing.assert_array_almost_equal(reconstructed, a)


def test_identity_matrix() -> None:
    a = np.eye(3, dtype=np.float64)
    l_matrix, u_matrix = lu_decomposition(a)
    np.testing.assert_array_almost_equal(l_matrix, np.eye(3))
    np.testing.assert_array_almost_equal(u_matrix, np.eye(3))


def test_lower_triangular_properties() -> None:
    a = np.array([[4, 3, 2], [6, 5, 1], [8, 7, 9]], dtype=np.float64)
    l_matrix, u_matrix = lu_decomposition(a)

    for i in range(l_matrix.shape[0]):
        assert l_matrix[i, i] == 1.0
        for j in range(i + 1, l_matrix.shape[1]):
            assert l_matrix[i, j] == 0.0

    for i in range(1, u_matrix.shape[0]):
        for j in range(i):
            assert u_matrix[i, j] == 0.0


@pytest.mark.parametrize(
    "a_data, b_data, expected",
    [
        ([[2, 3], [4, 7]], [8, 18], [1, 2]),
        ([[1, 2], [3, 4]], [5, 11], [1, 2]),
        ([[2, 1, 1], [4, -6, 0], [-2, 7, 2]], [5, -2, 9], [1, 1, 2]),
    ],
)
def test_lu_solve(
    a_data: Sequence[Sequence[float]],
    b_data: Sequence[float],
    expected: Sequence[float],
) -> None:
    a = np.array(a_data, dtype=np.float64)
    b = np.array(b_data, dtype=np.float64)
    expected_array = np.array(expected, dtype=np.float64)

    l_matrix, u_matrix = lu_decomposition(a)
    x = lu_solve(l_matrix, u_matrix, b)

    np.testing.assert_array_almost_equal(x, expected_array)
    np.testing.assert_array_almost_equal(a @ x, b)


def test_lu_invert() -> None:
    a = np.array([[2, 3], [4, 7]], dtype=np.float64)
    a_inv = lu_invert(a)

    identity = a @ a_inv
    np.testing.assert_array_almost_equal(identity, np.eye(2))

    a_inv_expected = np.array([[3.5, -1.5], [-2, 1]], dtype=np.float64)
    np.testing.assert_array_almost_equal(a_inv, a_inv_expected)


def test_lu_invert_3x3() -> None:
    a = np.array([[1, 2, 3], [0, 1, 4], [5, 6, 0]], dtype=np.float64)
    a_inv = lu_invert(a)

    identity = a @ a_inv
    np.testing.assert_array_almost_equal(identity, np.eye(3))


@click.group()
@click.option(
    "--log-level",
    "-l",
    default="INFO",
    type=click.Choice(
        ["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR"], case_sensitive=False
    ),
    help="Set logging level",
)
@click.pass_context
def cli(ctx: click.Context, log_level: str) -> None:
    ctx.ensure_object(dict)
    ctx.obj["log_level"] = log_level.upper()


@cli.command()
@click.option("--size", "-n", default=4, help="Size of square matrix")
@click.pass_context
def demo(ctx: click.Context, size: int) -> None:
    logger.remove()
    logger.add(
        lambda msg: click.echo(msg, err=True), level=ctx.obj["log_level"], colorize=True
    )

    logger.info("=" * 60)
    logger.info("LU DECOMPOSITION")
    logger.info("=" * 60)

    rng = random.Random(42)
    a_data = [[rng.randint(1, 9) for _ in range(size)] for _ in range(size)]
    a = np.array(a_data, dtype=np.float64)

    click.echo(f"\nMatrix A ({size}×{size}):{log_matrix(a, precision=1)}")

    logger.info("\nStarting LU decomposition...")
    l_matrix, u_matrix = lu_decomposition(a)

    logger.info("\n" + "=" * 60)
    logger.success("Decomposition complete!")
    logger.info("=" * 60)

    click.echo(f"\nMatrix L (Lower triangular):{log_matrix(l_matrix, precision=1)}")

    click.echo(f"\nMatrix U (Upper triangular):{log_matrix(u_matrix, precision=1)}")

    reconstructed = l_matrix @ u_matrix
    click.echo(f"\nVerification: L × U:{log_matrix(reconstructed, precision=1)}")

    if np.allclose(reconstructed, a):
        logger.success("\n✓ Verification passed: L × U = A")
    else:
        logger.error("\n✗ Verification failed: L × U ≠ A")


@cli.command()
@click.option("--size", "-n", default=3, help="Size of square matrix")
@click.option("--seed", "-s", default=42, help="Random seed")
@click.pass_context
def solve(ctx: click.Context, size: int, seed: int) -> None:
    logger.remove()
    logger.add(
        lambda msg: click.echo(msg, err=True), level=ctx.obj["log_level"], colorize=True
    )

    logger.info("=" * 60)
    logger.info("LU SOLVE - SYSTEM OF EQUATIONS")
    logger.info("=" * 60)

    rng = random.Random(seed)
    a_data = [[rng.randint(1, 9) for _ in range(size)] for _ in range(size)]
    a = np.array(a_data, dtype=np.float64)

    b_data = [rng.randint(1, 20) for _ in range(size)]
    b = np.array(b_data, dtype=np.float64)

    click.echo(f"\nMatrix A ({size}×{size}):{log_matrix(a, precision=1)}")
    click.echo(f"\nVector b: {b}")

    logger.info("\nSolving Ax = b using LU decomposition...")
    l_matrix, u_matrix = lu_decomposition(a)
    x = lu_solve(l_matrix, u_matrix, b)

    logger.info("\n" + "=" * 60)
    logger.success("Solution found!")
    logger.info("=" * 60)

    click.echo(f"\nSolution x: {x}")

    verification = a @ x
    click.echo(f"\nVerification: Ax = {verification}")

    if np.allclose(verification, b):
        logger.success("\n✓ Verification passed: Ax = b")
    else:
        logger.error("\n✗ Verification failed: Ax ≠ b")


@cli.command()
@click.option("--size", "-n", default=3, help="Size of square matrix")
@click.option("--seed", "-s", default=42, help="Random seed")
@click.pass_context
def invert(ctx: click.Context, size: int, seed: int) -> None:
    logger.remove()
    logger.add(
        lambda msg: click.echo(msg, err=True), level=ctx.obj["log_level"], colorize=True
    )

    logger.info("=" * 60)
    logger.info("LU INVERT - MATRIX INVERSION")
    logger.info("=" * 60)

    rng = random.Random(seed)
    a_data = [[rng.randint(1, 9) for _ in range(size)] for _ in range(size)]
    a = np.array(a_data, dtype=np.float64)

    click.echo(f"\nMatrix A ({size}×{size}):{log_matrix(a, precision=1)}")

    logger.info("\nComputing A^(-1) using LU decomposition...")
    a_inv = lu_invert(a)

    logger.info("\n" + "=" * 60)
    logger.success("Inverse computed!")
    logger.info("=" * 60)

    click.echo(f"\nMatrix A^(-1) ({size}×{size}):{log_matrix(a_inv, precision=3)}")

    identity = a @ a_inv
    click.echo(f"\nVerification: A × A^(-1):{log_matrix(identity, precision=3)}")

    if np.allclose(identity, np.eye(size)):
        logger.success("\n✓ Verification passed: A × A^(-1) = I")
    else:
        logger.error("\n✗ Verification failed: A × A^(-1) ≠ I")


@cli.command()
def test() -> None:
    logger.disable("")
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    cli()
