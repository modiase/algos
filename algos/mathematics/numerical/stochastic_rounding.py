#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.matplotlib -p python313Packages.numpy -p python313Packages.click -p python313Packages.pytest -p python313Packages.seaborn
"""
Stochastic (Probabilistic) Rounding vs Deterministic Rounding

Demonstrates error propagation in iterated affine transformations:
    x_{n+1} = round(A * x_n + B)

Stochastic rounding rounds to floor(x) with probability 1 - frac(x),
and to ceil(x) with probability frac(x), giving E[round(x)] = x.

This unbiased property prevents systematic error accumulation that
plagues deterministic rounding in iterative computations.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from decimal import Decimal, getcontext
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pytest
import seaborn as sns

getcontext().prec = 50


@dataclass
class IterationResult:
    """Results from running iterated affine transformation."""

    exact: npt.NDArray[np.float64]
    deterministic: npt.NDArray[np.float64]
    stochastic: npt.NDArray[np.float64]


def high_precision_affine_iterate(
    x0: float, a: float, b: float, n_iterations: int
) -> list[Decimal]:
    """Compute exact iteration using high-precision Decimal arithmetic."""
    x = Decimal(str(x0))
    a_dec = Decimal(str(a))
    b_dec = Decimal(str(b))

    trajectory = [x]
    for _ in range(n_iterations):
        x = a_dec * x + b_dec
        trajectory.append(x)

    return trajectory


def deterministic_round_to_precision(x: float, decimals: int) -> float:
    """Round to fixed decimal places using standard rounding."""
    factor = 10**decimals
    return np.round(x * factor) / factor


def stochastic_round_to_precision(
    x: float, decimals: int, rng: np.random.Generator
) -> float:
    """
    Stochastically round to fixed decimal places.

    Rounds down with probability (1 - fractional_part),
    rounds up with probability fractional_part.
    """
    factor = 10**decimals
    scaled = x * factor
    floor_val = np.floor(scaled)
    frac = scaled - floor_val

    if rng.random() < frac:
        return (floor_val + 1) / factor
    return floor_val / factor


def iterate_affine_transform(
    x0: float,
    a: float,
    b: float,
    n_iterations: int,
    decimals: int,
    rng: np.random.Generator | None = None,
) -> IterationResult:
    """
    Run iterated affine transformation with different rounding strategies.

    Computes x_{n+1} = round(a * x_n + b) using:
    - High-precision arithmetic (reference)
    - Deterministic rounding
    - Stochastic rounding
    """
    if rng is None:
        rng = np.random.default_rng()

    exact_decimal = high_precision_affine_iterate(x0, a, b, n_iterations)
    exact = np.array([float(d) for d in exact_decimal])

    det_trajectory = [x0]
    x_det = x0
    for _ in range(n_iterations):
        x_det = deterministic_round_to_precision(a * x_det + b, decimals)
        det_trajectory.append(x_det)

    stoch_trajectory = [x0]
    x_stoch = x0
    for _ in range(n_iterations):
        x_stoch = stochastic_round_to_precision(a * x_stoch + b, decimals, rng)
        stoch_trajectory.append(x_stoch)

    return IterationResult(
        exact=exact,
        deterministic=np.array(det_trajectory),
        stochastic=np.array(stoch_trajectory),
    )


def run_trials(
    x0: float,
    a: float,
    b: float,
    n_iterations: int,
    decimals: int,
    n_trials: int,
    seed: int | None = None,
) -> dict[str, npt.NDArray[np.float64]]:
    """Run multiple trials gathering error statistics."""
    rng = np.random.default_rng(seed)

    det_final_errors = np.zeros(n_trials)
    stoch_final_errors = np.zeros(n_trials)
    det_max_errors = np.zeros(n_trials)
    stoch_max_errors = np.zeros(n_trials)

    for trial in range(n_trials):
        result = iterate_affine_transform(x0, a, b, n_iterations, decimals, rng)

        det_errors = result.deterministic - result.exact
        stoch_errors = result.stochastic - result.exact

        det_final_errors[trial] = det_errors[-1]
        stoch_final_errors[trial] = stoch_errors[-1]
        det_max_errors[trial] = np.max(np.abs(det_errors))
        stoch_max_errors[trial] = np.max(np.abs(stoch_errors))

    return {
        "det_final": det_final_errors,
        "stoch_final": stoch_final_errors,
        "det_max": det_max_errors,
        "stoch_max": stoch_max_errors,
    }


def plot_comparison(
    x0: float,
    a: float,
    b: float,
    n_iterations: int,
    decimals: int,
    n_trials: int,
    output_path: Path,
    seed: int | None = None,
) -> None:
    """Generate comparison plots for stochastic vs deterministic rounding."""
    sns.set_theme(style="whitegrid", palette="deep")
    _, axes = plt.subplots(2, 2, figsize=(14, 10))

    rng = np.random.default_rng(seed)

    # Top-left: Single trajectory comparison
    ax_traj = axes[0, 0]
    result = iterate_affine_transform(x0, a, b, n_iterations, decimals, rng)
    steps = np.arange(n_iterations + 1)

    ax_traj.plot(steps, result.exact, label="Exact", color="black", linewidth=2)
    ax_traj.plot(
        steps, result.deterministic, label="Deterministic", color="tab:red", alpha=0.8
    )
    ax_traj.plot(
        steps, result.stochastic, label="Stochastic", color="tab:blue", alpha=0.8
    )
    ax_traj.set_xlabel("Iteration")
    ax_traj.set_ylabel("Value")
    ax_traj.set_title(f"Trajectory: x → {a}x + {b} (rounded to {decimals} d.p.)")
    ax_traj.legend()

    # Top-right: Error trajectory
    ax_err = axes[0, 1]
    det_err = result.deterministic - result.exact
    stoch_err = result.stochastic - result.exact

    ax_err.plot(steps, det_err, label="Deterministic", color="tab:red", alpha=0.8)
    ax_err.plot(steps, stoch_err, label="Stochastic", color="tab:blue", alpha=0.8)
    ax_err.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax_err.set_xlabel("Iteration")
    ax_err.set_ylabel("Error (rounded - exact)")
    ax_err.set_title("Error trajectory (single trial)")
    ax_err.legend()

    # Bottom-left: Final error distribution
    ax_dist = axes[1, 0]
    mc_results = run_trials(x0, a, b, n_iterations, decimals, n_trials * 2, seed)

    ax_dist.hist(
        mc_results["det_final"],
        bins=30,
        alpha=0.6,
        label=f"Deterministic (μ={np.mean(mc_results['det_final']):.4f})",
        color="tab:red",
    )
    ax_dist.hist(
        mc_results["stoch_final"],
        bins=30,
        alpha=0.6,
        label=f"Stochastic (μ={np.mean(mc_results['stoch_final']):.4f})",
        color="tab:blue",
    )
    ax_dist.axvline(x=0, color="black", linestyle="--", linewidth=2)
    ax_dist.set_xlabel("Final error")
    ax_dist.set_ylabel("Frequency")
    ax_dist.set_title(f"Final error distribution ({n_trials * 2} trials)")
    ax_dist.legend()

    # Bottom-right: RMSE growth with iteration count
    ax_growth = axes[1, 1]
    iteration_counts = [10, 25, 50, 100, 200, 500]
    det_rmse = []
    stoch_rmse = []

    for n_iter in iteration_counts:
        stats = run_trials(x0, a, b, n_iter, decimals, n_trials, seed)
        det_rmse.append(np.sqrt(np.mean(stats["det_final"] ** 2)))
        stoch_rmse.append(np.sqrt(np.mean(stats["stoch_final"] ** 2)))

    ax_growth.plot(
        iteration_counts, det_rmse, "o-", label="Deterministic", color="tab:red"
    )
    ax_growth.plot(
        iteration_counts, stoch_rmse, "s-", label="Stochastic", color="tab:blue"
    )
    ax_growth.set_xlabel("Number of iterations")
    ax_growth.set_ylabel("RMSE")
    ax_growth.set_title("RMSE growth with iteration count")
    ax_growth.set_xscale("log")
    ax_growth.set_yscale("log")
    ax_growth.legend()

    plt.suptitle(
        f"Iterated Affine Transform: x → {a}x + {b}\n"
        f"Stochastic vs Deterministic Rounding ({decimals} decimal places)",
        fontsize=14,
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


@pytest.mark.parametrize("x", [0.0, 1.0, 5.0, -3.0])
def test_deterministic_round_exact_values(x: float) -> None:
    """Values with no fractional part should round to themselves."""
    assert deterministic_round_to_precision(x, 2) == x


def test_stochastic_round_unbiased() -> None:
    """Mean of many stochastic rounds should converge to true value."""
    rng = np.random.default_rng(42)
    x = 3.456
    n_samples = 10000
    results = [stochastic_round_to_precision(x, 1, rng) for _ in range(n_samples)]
    mean_result = np.mean(results)
    assert abs(mean_result - x) < 0.02


def test_stochastic_round_bounds() -> None:
    """Stochastic round should only return adjacent representable values."""
    rng = np.random.default_rng(42)
    x = 5.34
    decimals = 1
    for _ in range(100):
        result = stochastic_round_to_precision(x, decimals, rng)
        assert result in (5.3, 5.4)


def test_high_precision_matches_float() -> None:
    """High precision should match float64 for simple cases."""
    trajectory = high_precision_affine_iterate(1.0, 1.1, 0.5, 10)
    x = 1.0
    for i in range(10):
        x = 1.1 * x + 0.5
        assert abs(float(trajectory[i + 1]) - x) < 1e-10


def test_deterministic_accumulates_bias() -> None:
    """Deterministic rounding should accumulate systematic error."""
    stats = run_trials(1.0, 1.05, 0.33, 100, 1, 100, seed=42)
    det_bias = abs(np.mean(stats["det_final"]))
    stoch_bias = abs(np.mean(stats["stoch_final"]))

    assert det_bias > stoch_bias * 2


cli = click.Group()


@cli.command()
@click.option("--x0", default=1.0, help="Initial value")
@click.option("--a", default=0.95, help="Multiplicative factor")
@click.option("--b", default=0.73, help="Additive bias")
@click.option("--iterations", "-n", default=200, help="Number of iterations")
@click.option("--decimals", "-d", default=1, help="Decimal places for rounding")
@click.option("--trials", "-t", default=100, help="Number of Monte Carlo trials")
@click.option("--seed", "-s", default=42, help="Random seed")
@click.option("--output", "-o", default=None, help="Output path for plot")
def demo(
    x0: float,
    a: float,
    b: float,
    iterations: int,
    decimals: int,
    trials: int,
    seed: int,
    output: str | None,
) -> None:
    """Run iterated affine transformation demo with error analysis."""
    output_path = (
        Path(output)
        if output
        else Path(tempfile.gettempdir()) / "stochastic_rounding.png"
    )

    click.echo("Iterated Affine Transform: x → Ax + B with Rounding")
    click.echo("=" * 55)
    click.echo(f"Transform: x → {a}x + {b}")
    click.echo(f"Initial value: {x0}")
    click.echo(f"Rounding precision: {decimals} decimal places")
    click.echo(f"Iterations: {iterations}, Trials: {trials}\n")

    stats = run_trials(x0, a, b, iterations, decimals, trials, seed)

    click.echo("Final Error Statistics:")
    click.echo("-" * 55)
    click.echo(f"{'Method':<15} {'RMSE':>12} {'Bias':>12} {'Max |Error|':>12}")
    click.echo("-" * 55)

    det_rmse = np.sqrt(np.mean(stats["det_final"] ** 2))
    det_bias = np.mean(stats["det_final"])
    det_max = np.mean(stats["det_max"])
    click.echo(
        f"{'Deterministic':<15} {det_rmse:>12.6f} {det_bias:>+12.6f} {det_max:>12.6f}"
    )

    stoch_rmse = np.sqrt(np.mean(stats["stoch_final"] ** 2))
    stoch_bias = np.mean(stats["stoch_final"])
    stoch_max = np.mean(stats["stoch_max"])
    click.echo(
        f"{'Stochastic':<15} {stoch_rmse:>12.6f} {stoch_bias:>+12.6f} {stoch_max:>12.6f}"
    )

    click.echo("\nKey observations:")
    click.echo("• Deterministic rounding accumulates systematic bias")
    click.echo("• Stochastic rounding remains unbiased (mean error ≈ 0)")
    click.echo("• Stochastic has higher variance but no drift")

    plot_comparison(x0, a, b, iterations, decimals, trials, output_path, seed)
    click.echo(f"\nPlot saved to: {output_path}")


@cli.command("test")
def run_tests() -> None:
    """Run pytest on this module."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "-p", "no:cacheprovider"],
        cwd=Path(__file__).parent,
    )
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    cli()
