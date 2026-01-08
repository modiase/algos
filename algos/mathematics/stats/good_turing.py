#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.matplotlib -p python313Packages.numpy -p python313Packages.click -p python313Packages.pytest -p python313Packages.seaborn
from __future__ import annotations

import tempfile
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pytest
import seaborn as sns


def zipf_distribution(vocab_size: int) -> npt.NDArray[np.floating]:
    """Generate a Zipf distribution over vocab_size items."""
    ranks = np.arange(1, vocab_size + 1)
    probs = 1.0 / ranks
    return probs / probs.sum()


def geometric_distribution(
    vocab_size: int, p: float = 0.01
) -> npt.NDArray[np.floating]:
    """Generate a truncated geometric distribution over vocab_size items."""
    indices = np.arange(vocab_size)
    probs = (1 - p) ** indices * p
    return probs / probs.sum()


def uniform_distribution(vocab_size: int) -> npt.NDArray[np.floating]:
    """Generate a uniform distribution over vocab_size items."""
    return np.ones(vocab_size) / vocab_size


def sample_from_distribution(
    probs: Sequence[float] | npt.NDArray[np.floating], n_samples: int
) -> npt.NDArray[np.intp]:
    """Draw n_samples from a categorical distribution defined by probs."""
    return np.random.choice(len(probs), size=n_samples, p=probs)


def compute_true_missing_mass(
    probs: Sequence[float] | npt.NDArray[np.floating], observed: set[int]
) -> float:
    """Compute the true missing mass (probability of unseen items)."""
    return sum(probs[i] for i in range(len(probs)) if i not in observed)


def good_turing_missing_mass(samples: Sequence[int] | npt.NDArray[np.intp]) -> float:
    """Estimate missing mass using Good-Turing: N1/n where N1 is singleton count."""
    n = len(samples)
    if n == 0:
        return 0.0
    counts = Counter(samples)
    n1 = sum(1 for c in counts.values() if c == 1)
    return n1 / n


DistributionFn = Callable[[int], npt.NDArray[np.floating]]


SimulationResult = tuple[
    Sequence[float],  # gt_means
    Sequence[float],  # gt_stds
    Sequence[float],  # true_means
    Sequence[float],  # true_stds
    Sequence[float],  # risks (MSE)
]


def run_simulation(
    vocab_size: int,
    sample_sizes: Sequence[int],
    n_trials: int,
    distribution_fn: DistributionFn,
) -> SimulationResult:
    """Run Good-Turing simulation across multiple sample sizes and trials."""
    probs = distribution_fn(vocab_size)

    gt_means, gt_stds = [], []
    true_means, true_stds = [], []
    risks = []

    for n in sample_sizes:
        gt_estimates = []
        true_values = []

        for _ in range(n_trials):
            samples = sample_from_distribution(probs, n)
            observed = set(samples)

            gt_estimates.append(good_turing_missing_mass(samples))
            true_values.append(compute_true_missing_mass(probs, observed))

        gt_arr = np.array(gt_estimates)
        true_arr = np.array(true_values)

        gt_means.append(np.mean(gt_arr))
        gt_stds.append(np.std(gt_arr))
        true_means.append(np.mean(true_arr))
        true_stds.append(np.std(true_arr))
        risks.append(np.mean((gt_arr - true_arr) ** 2))

    return gt_means, gt_stds, true_means, true_stds, risks


def plot_distribution_comparison(
    sample_sizes: Sequence[int],
    results: dict[str, SimulationResult],
    output_path: Path,
) -> None:
    """Plot Good-Turing performance comparison across distributions."""
    sns.set_theme(style="whitegrid", palette="deep")

    _, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = sns.color_palette("husl", len(results))

    ax_est = axes[0, 0]
    for (name, (gt_means, _, true_means, _, _)), color in zip(results.items(), colors):
        ax_est.plot(
            sample_sizes,
            true_means,
            marker="o",
            linestyle="-",
            color=color,
            label=f"{name} (True)",
            alpha=0.8,
        )
        ax_est.plot(
            sample_sizes,
            gt_means,
            marker="s",
            linestyle="--",
            color=color,
            label=f"{name} (GT)",
            alpha=0.8,
        )
    ax_est.set_xlabel("Sample Size")
    ax_est.set_ylabel("Missing Mass")
    ax_est.set_title("True vs Good-Turing Estimates")
    ax_est.set_xscale("log")
    ax_est.legend(fontsize=8, loc="upper right")

    ax_err = axes[0, 1]
    for (name, (gt_means, _, true_means, _, _)), color in zip(results.items(), colors):
        gt_arr = np.array(gt_means)
        true_arr = np.array(true_means)
        rel_errors = np.abs(gt_arr - true_arr) / (true_arr + 1e-10) * 100
        ax_err.plot(
            sample_sizes, rel_errors, marker="o", color=color, label=name, linewidth=2
        )
    ax_err.set_xlabel("Sample Size")
    ax_err.set_ylabel("Relative Error (%)")
    ax_err.set_title("Good-Turing Estimation Error by Distribution")
    ax_err.set_xscale("log")
    ax_err.legend()
    ax_err.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    ax_risk = axes[1, 0]
    for (name, (_, _, _, _, risks)), color in zip(results.items(), colors):
        ax_risk.plot(
            sample_sizes, risks, marker="o", color=color, label=name, linewidth=2
        )
    ax_risk.set_xlabel("Sample Size")
    ax_risk.set_ylabel("Risk (MSE)")
    ax_risk.set_title("Good-Turing Estimation Risk")
    ax_risk.set_xscale("log")
    ax_risk.set_yscale("log")
    ax_risk.legend()

    ax_dist = axes[1, 1]
    vocab_preview = 100
    distributions: dict[str, DistributionFn] = {
        "Zipf": zipf_distribution,
        "Geometric": geometric_distribution,
        "Uniform": uniform_distribution,
    }
    for (name, dist_fn), color in zip(distributions.items(), colors):
        probs = dist_fn(vocab_preview)
        ax_dist.plot(range(vocab_preview), probs, color=color, label=name, linewidth=2)
    ax_dist.set_xlabel("Item Rank")
    ax_dist.set_ylabel("Probability")
    ax_dist.set_title("Distribution Shapes (first 100 items)")
    ax_dist.legend()
    ax_dist.set_yscale("log")

    plt.suptitle(
        "Good-Turing Missing Mass Estimation: Distribution Comparison",
        fontsize=14,
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


@pytest.mark.parametrize(
    "samples, expected_range",
    [
        ([0, 1, 2, 3, 4], (0.9, 1.1)),
        ([0, 0, 1, 1, 2, 2], (0.0, 0.1)),
        ([0, 0, 0, 1, 2, 3], (0.4, 0.6)),
    ],
)
def test_good_turing_missing_mass(
    samples: list[int], expected_range: tuple[float, float]
) -> None:
    result = good_turing_missing_mass(samples)
    assert expected_range[0] <= result <= expected_range[1]


def test_zipf_distribution_sums_to_one() -> None:
    probs = zipf_distribution(1000)
    assert abs(sum(probs) - 1.0) < 1e-10


def test_geometric_distribution_sums_to_one() -> None:
    probs = geometric_distribution(1000)
    assert abs(sum(probs) - 1.0) < 1e-10


def test_uniform_distribution_sums_to_one() -> None:
    probs = uniform_distribution(1000)
    assert abs(sum(probs) - 1.0) < 1e-10


def test_true_missing_mass_with_all_observed() -> None:
    probs = [0.25, 0.25, 0.25, 0.25]
    observed = {0, 1, 2, 3}
    assert compute_true_missing_mass(probs, observed) == 0.0


def test_true_missing_mass_with_none_observed() -> None:
    probs = [0.25, 0.25, 0.25, 0.25]
    observed: set[int] = set()
    assert abs(compute_true_missing_mass(probs, observed) - 1.0) < 1e-10


cli = click.Group()


@cli.command()
@click.option("--vocab-size", "-v", default=10000, help="Vocabulary size")
@click.option("--n-trials", "-t", default=50, help="Number of trials per sample size")
@click.option(
    "--output",
    "-o",
    default=None,
    help="Output path for plot (defaults to $TMPDIR/good_turing_simulation.png)",
)
def demo(vocab_size: int, n_trials: int, output: str | None) -> None:
    """Run Good-Turing missing mass estimation simulation comparing distributions."""
    output_path = (
        Path(output)
        if output
        else Path(tempfile.gettempdir()) / "good_turing_simulation.png"
    )

    sample_sizes = [50, 100, 200, 500, 1000, 2000, 5000, 10000]

    distributions: dict[str, DistributionFn] = {
        "Zipf": zipf_distribution,
        "Geometric": geometric_distribution,
        "Uniform": uniform_distribution,
    }

    click.echo(f"Running simulation with vocab_size={vocab_size}, trials={n_trials}")
    click.echo(f"Comparing distributions: {', '.join(distributions.keys())}\n")

    results: dict[str, SimulationResult] = {}

    for name, dist_fn in distributions.items():
        click.echo(f"Simulating {name} distribution...")
        results[name] = run_simulation(vocab_size, sample_sizes, n_trials, dist_fn)

    click.echo("\nResults Summary (Relative Error %):")
    header = f"{'Sample Size':>12}" + "".join(f"{name:>12}" for name in distributions)
    click.echo(header)
    click.echo("-" * len(header))

    for i, n in enumerate(sample_sizes):
        row = f"{n:>12}"
        for name in distributions:
            gt_mean = results[name][0][i]
            true_mean = results[name][2][i]
            rel_err = abs(gt_mean - true_mean) / (true_mean + 1e-10) * 100
            row += f"{rel_err:>11.2f}%"
        click.echo(row)

    plot_distribution_comparison(sample_sizes, results, output_path)
    click.echo(f"\nPlot saved to: {output_path}")


@cli.command("test")
def run_tests() -> None:
    """Run pytest on this module."""
    import sys
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "-p", "no:cacheprovider"],
        cwd=Path(__file__).parent,
    )
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    cli()
