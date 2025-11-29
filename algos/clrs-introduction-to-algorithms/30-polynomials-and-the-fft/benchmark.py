#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.click -p python313Packages.numpy -p python313Packages.jax -p python313Packages.jaxlib -p python313Packages.seaborn -p python313Packages.matplotlib -p python313Packages.pandas -p python313Packages.pytest
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import time
from pathlib import Path

import click
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append(str(Path(__file__).parent))


def load_module(module_name: str, file_path: Path):
    """Load a module from a file path and register it in sys.modules for pickling."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise RuntimeError("Spec is None")
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("Loader is None")
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


parent_dir = Path(__file__).parent
dft = load_module("dft", parent_dir / "dft.py")
iterative_fft = load_module("iterative_fft", parent_dir / "iterative-fft.py")
recursive_fft = load_module("recursive_fft", parent_dir / "recursive-fft.py")
parallel_fft = load_module("parallel_fft", parent_dir / "parallel-iterative-fft.py")


def benchmark_single_run(args):
    """Run a single benchmark (for parallel execution)."""
    n, algorithm, trial_seed = args
    np.random.seed(trial_seed)
    coeffs_np = np.random.randn(n)

    if algorithm == "Naive DFT":
        coeffs_jax = jnp.array(coeffs_np)
        start = time.perf_counter()
        _ = dft.dft(coeffs_jax, n)
        elapsed = time.perf_counter() - start
        return {"n": n, "Algorithm": algorithm, "Time (s)": elapsed}
    elif algorithm == "Recursive FFT":
        coeffs_jax = jnp.array(coeffs_np)
        start = time.perf_counter()
        _ = recursive_fft.fft(coeffs_jax, n)
        elapsed = time.perf_counter() - start
        return {"n": n, "Algorithm": algorithm, "Time (s)": elapsed}
    elif algorithm == "Iterative FFT":
        start = time.perf_counter()
        _ = iterative_fft.fft(coeffs_np, n)
        elapsed = time.perf_counter() - start
        return {"n": n, "Algorithm": algorithm, "Time (s)": elapsed}
    elif algorithm == "Parallel FFT":
        start = time.perf_counter()
        _ = parallel_fft.fft(coeffs_np, n)
        elapsed = time.perf_counter() - start
        return {"n": n, "Algorithm": algorithm, "Time (s)": elapsed}
    return None


def run_benchmarks(
    start_n: int = 8,
    ratio: float = 2.0,
    max_time: float = 10.0,
    max_n: int = 10000,
    trials: int = 5,
) -> pd.DataFrame:
    """
    Run benchmarks with geometric progression, dropping algorithms when they exceed max_time.
    """
    algorithms = ["Naive DFT", "Recursive FFT", "Iterative FFT", "Parallel FFT"]
    active_algorithms = set(algorithms)
    results = []

    n = start_n
    while active_algorithms:
        if max_n > 0 and n > max_n:
            click.echo(f"\nReached max_n={max_n}, stopping benchmark")
            break

        click.echo(f"\nBenchmarking n={n}")

        for algorithm in list(active_algorithms):
            times = []
            for trial in range(trials):
                trial_seed = hash((n, trial)) % (2**32)
                result = benchmark_single_run((n, algorithm, trial_seed))
                if result:
                    times.append(result["Time (s)"])
                    results.append(result)

            avg_time = np.mean(times) if times else 0
            click.echo(f"  {algorithm}: {avg_time:.4f}s")

            if avg_time > max_time:
                click.echo(f"  -> Dropped {algorithm} (exceeded {max_time}s)")
                active_algorithms.discard(algorithm)

        n = int(n * ratio)

    return pd.DataFrame(results)


def plot_benchmarks(df: pd.DataFrame, output: str, max_time: float = 10.0) -> None:
    """Plot benchmark results using seaborn."""
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    sns.lineplot(
        data=df,
        x="n",
        y="Time (s)",
        hue="Algorithm",
        marker="o",
        ax=ax,
        errorbar=None,
    )

    ax.axhline(y=max_time, color="red", linestyle="--", linewidth=1.5, alpha=0.7)

    ax.set_xlabel("Input Size (n)", fontsize=12)
    ax.set_ylabel("Time (seconds)", fontsize=12)
    ax.set_title("FFT Performance Comparison", fontsize=14)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(loc="lower right", frameon=True, fancybox=False)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(output, dpi=150, facecolor="white")
    click.echo(f"Saved benchmark plot to {output}")


cli = click.Group()


@cli.command()
@click.option("--start-n", "-s", default=8, help="Starting size (default: 8)", type=int)
@click.option(
    "--ratio", "-r", default=2.0, help="Geometric progression ratio (default: 2.0)"
)
@click.option(
    "--max-time",
    "-m",
    default=10.0,
    help="Maximum time before dropping algorithm (default: 10.0s)",
)
@click.option(
    "--max-n",
    "-n",
    default=10000,
    help="Maximum size to benchmark (default: 10000, -1 for unlimited)",
    type=int,
)
@click.option(
    "--trials", "-t", default=5, help="Number of trials per size for averaging"
)
@click.option("--output", "-o", default=None, help="Output file for visualization")
def run(
    start_n: int,
    ratio: float,
    max_time: float,
    max_n: int,
    trials: int,
    output: str | None,
) -> None:
    """Run FFT benchmarks with geometric progression and plot results."""
    if output is None:
        output = os.path.join(tempfile.gettempdir(), "fft_benchmark.png")

    max_n_str = "unlimited" if max_n == -1 else str(max_n)
    click.echo(
        f"Running benchmarks: start_n={start_n}, ratio={ratio}, max_time={max_time}s, max_n={max_n_str}, trials={trials}"
    )

    df = run_benchmarks(start_n, ratio, max_time, max_n, trials)

    click.echo("\nBenchmark Results Summary:")
    click.echo(df.groupby(["n", "Algorithm"])["Time (s)"].mean().to_string())
    click.echo()

    plot_benchmarks(df, output, max_time)
    os.system(f"open {output}")


if __name__ == "__main__":
    run()
