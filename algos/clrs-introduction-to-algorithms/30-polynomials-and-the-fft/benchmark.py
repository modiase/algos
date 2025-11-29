#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.click -p python313Packages.numpy -p python313Packages.jax -p python313Packages.jaxlib -p python313Packages.seaborn -p python313Packages.matplotlib -p python313Packages.pandas -p python313Packages.pytest
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import click
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append(str(Path(__file__).parent))


def load_module(module_name: str, file_path: Path):
    """Load a module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise RuntimeError("Spec is None")
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("Loader is None")
    spec.loader.exec_module(module)
    return module


parent_dir = Path(__file__).parent
dft = load_module("dft", parent_dir / "dft.py")
fft = load_module("fft", parent_dir / "fft.py")
recursive_fft = load_module("recursive_fft", parent_dir / "recursive-fft.py")


def benchmark_single_run(args):
    """Run a single benchmark (for parallel execution)."""
    n, algorithm, trial_seed = args
    np.random.seed(trial_seed)
    coeffs_np = np.random.randn(n)

    if algorithm == "Naive DFT" and n <= 1000:
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
        _ = fft.fft(coeffs_np, n)
        elapsed = time.perf_counter() - start
        return {"n": n, "Algorithm": algorithm, "Time (s)": elapsed}
    return None


def run_benchmarks(sizes: list[int], trials: int = 5) -> pd.DataFrame:
    """Run benchmarks for all implementations across different sizes."""
    tasks = []
    for n in sizes:
        for trial in range(trials):
            trial_seed = hash((n, trial)) % (2**32)
            if n <= 1000:
                tasks.append((n, "Naive DFT", trial_seed))
            tasks.append((n, "Recursive FFT", trial_seed))
            tasks.append((n, "Iterative FFT", trial_seed))

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(benchmark_single_run, tasks))

    return pd.DataFrame([r for r in results if r is not None])


def plot_benchmarks(df: pd.DataFrame, output: str) -> None:
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
@click.option(
    "--sizes",
    "-s",
    default="3,10,30,100,300,1000",
    help="Comma-separated list of sizes to benchmark",
)
@click.option(
    "--trials", "-t", default=5, help="Number of trials per size for averaging"
)
@click.option("--output", "-o", default=None, help="Output file for visualization")
def run(sizes: str, trials: int, output: str | None) -> None:
    """Run FFT benchmarks and plot results."""
    size_list = [int(s.strip()) for s in sizes.split(",")]

    if output is None:
        output = os.path.join(tempfile.gettempdir(), "fft_benchmark.png")

    click.echo(f"Running benchmarks for sizes: {size_list}")
    click.echo(f"Trials per size: {trials}")
    click.echo()

    df = run_benchmarks(size_list, trials)

    click.echo("\nBenchmark Results Summary:")
    click.echo(df.groupby(["n", "Algorithm"])["Time (s)"].mean().to_string())
    click.echo()

    plot_benchmarks(df, output)
    os.system(f"open {output}")


if __name__ == "__main__":
    run()
