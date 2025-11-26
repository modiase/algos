#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.loguru -p python313Packages.pytest -p python313Packages.click -p python313Packages.numpy -p python313Packages.seaborn -p python313Packages.matplotlib
from __future__ import annotations

import os
import tempfile

import click
import matplotlib.pyplot as plt
import numpy as np
import pytest
import seaborn as sns
from numpy.typing import NDArray


def whiten(data: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """
    ZCA (Zero-phase Component Analysis) whitening, also known as Mahalanobis whitening.

    Whitening is a linear transformation that decorrelates data and scales each dimension
    to unit variance, resulting in an identity covariance matrix. This is useful for
    algorithms that assume features are independent and identically distributed.

    This implementation uses ZCA whitening, which produces whitened data that remains
    closest to the original data in terms of Euclidean distance. Unlike PCA whitening,
    ZCA rotates back to the original feature space after scaling.

    Approach:
    1. Center the data by subtracting the mean
    2. Compute the covariance matrix C = (1/n) X^T X
    3. Eigendecompose C = V D V^T
    4. Construct whitening matrix W = V D^(-1/2) V^T
    5. Transform data: X_whitened = X_centered W^T

    The whitening matrix W transforms the data such that the whitened covariance
    becomes the identity: W^T C W = I
    """
    centered = data - np.mean(data, axis=0)
    cov = np.cov(centered.T)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    whitening_matrix = (
        eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
    )

    return centered @ whitening_matrix.T, whitening_matrix, cov


def plot_projections(original: NDArray, whitened: NDArray, output: str) -> None:
    sns.set_style("whitegrid")
    _, axes = plt.subplots(2, 3, figsize=(18, 12))

    original_limit = np.max(np.abs(original)) * 1.1
    whitened_limit = 3.5

    for idx, (i, j, label_i, label_j) in enumerate(
        [
            (0, 1, "X", "Y"),
            (0, 2, "X", "Z"),
            (1, 2, "Y", "Z"),
        ]
    ):
        axes[0, idx].scatter(original[:, i], original[:, j], alpha=0.6, s=20)
        axes[0, idx].set_xlabel(label_i, fontsize=12)
        axes[0, idx].set_ylabel(label_j, fontsize=12)
        axes[0, idx].set_title(
            f"Original: {label_i}-{label_j}", fontsize=14, fontweight="bold"
        )
        axes[0, idx].set_xlim(-original_limit, original_limit)
        axes[0, idx].set_ylim(-original_limit, original_limit)
        axes[0, idx].set_aspect("equal", adjustable="box")
        axes[0, idx].grid(True, alpha=0.3)

        axes[1, idx].scatter(
            whitened[:, i], whitened[:, j], alpha=0.6, s=20, color="orange"
        )
        axes[1, idx].set_xlabel(label_i, fontsize=12)
        axes[1, idx].set_ylabel(label_j, fontsize=12)
        axes[1, idx].set_title(
            f"Whitened: {label_i}-{label_j}", fontsize=14, fontweight="bold"
        )
        axes[1, idx].set_xlim(-whitened_limit, whitened_limit)
        axes[1, idx].set_ylim(-whitened_limit, whitened_limit)
        axes[1, idx].set_aspect("equal", adjustable="box")
        axes[1, idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    click.echo(f"Saved visualization to {output}")


@pytest.mark.parametrize(
    "seed, n, d",
    [
        (42, 100, 3),
        (123, 50, 3),
    ],
)
def test_whiten(seed: int, n: int, d: int) -> None:
    np.random.seed(seed)
    data = np.random.randn(n, d) @ np.random.randn(d, d)
    whitened, _, _ = whiten(data)
    whitened_cov = np.cov(whitened.T)
    assert np.allclose(whitened_cov, np.eye(d), atol=1e-10)


cli = click.Group()


@cli.command()
@click.option("--seed", "-s", default=42, help="Random seed for data generation")
@click.option("--n", "-n", default=100, help="Number of samples to generate")
@click.option("--output", "-o", default=None, help="Output file for visualization")
def demo(seed: int, n: int, output: str | None) -> None:
    np.random.seed(seed)
    d = 3

    if output is None:
        output = os.path.join(tempfile.gettempdir(), f"whitening_seed{seed}_n{n}.png")

    click.echo(f"Generating {n} samples of {d}-dimensional data with seed={seed}")

    data = np.random.randn(n, d) @ np.random.randn(d, d)

    click.echo("Computing whitening transformation...")
    whitened, whitening_matrix, original_cov = whiten(data)
    whitened_cov = np.cov(whitened.T)

    click.echo("\nOriginal Covariance Matrix:")
    click.echo(original_cov)

    click.echo("\nWhitening Matrix:")
    click.echo(whitening_matrix)

    click.echo("\nWhitened Covariance Matrix (should be identity):")
    click.echo(whitened_cov)

    click.echo(
        f"\nMax deviation from identity: {np.max(np.abs(whitened_cov - np.eye(d))):.2e}"
    )

    plot_projections(data, whitened, output)

    os.system(f"open {output}")


@cli.command("test")
def run_tests() -> None:
    pytest.main([__file__])


if __name__ == "__main__":
    cli()
