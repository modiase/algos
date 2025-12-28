#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.pytest -p python313Packages.click -p python313Packages.numpy
"""
Demonstrate splay tree behavior under different access patterns.

This script explores how splay trees function as a cache by analyzing
access patterns and tree statistics. It shows that frequently accessed
values tend to stay near the root with lower average depths.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import click
import numpy as np
import pytest

from splay import get_depth, insert, is_root, search


@dataclass(frozen=True)
class ValueStats:
    """Statistics for a single value across all accesses."""

    value: int
    access_count: int
    hit_count: int
    total_cost: int

    @property
    def avg_cost(self) -> float:
        return self.total_cost / self.access_count if self.access_count > 0 else 0.0

    @property
    def hit_rate(self) -> float:
        return self.hit_count / self.access_count if self.access_count > 0 else 0.0


@dataclass(frozen=True)
class OverallStats:
    """Overall statistics across all accesses."""

    total_accesses: int
    total_hits: int
    total_cost: int

    @property
    def avg_cost(self) -> float:
        return self.total_cost / self.total_accesses if self.total_accesses > 0 else 0.0

    @property
    def hit_rate(self) -> float:
        return self.total_hits / self.total_accesses if self.total_accesses > 0 else 0.0


def generate_probabilities(num_values: int, seed: int) -> np.ndarray:
    """
    Generate access probabilities using softmax over random logits.

    Returns array of probabilities summing to 1.0.
    """
    rng = np.random.default_rng(seed)
    logits = rng.standard_normal(num_values)
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / exp_logits.sum()


def run_access_simulation(
    values: list[int], probabilities: np.ndarray, num_accesses: int, seed: int
) -> tuple[OverallStats, dict[int, ValueStats]]:
    """
    Simulate random accesses and collect statistics.

    Returns (overall_stats, per_value_stats).
    """
    rng = np.random.default_rng(seed)

    root = None
    for val in values:
        root = insert(root, val)

    total_cost = 0
    total_hits = 0
    value_access_counts: dict[int, int] = defaultdict(int)
    value_hit_counts: dict[int, int] = defaultdict(int)
    value_cost_sums: dict[int, int] = defaultdict(int)

    for _ in range(num_accesses):
        accessed_value = rng.choice(values, p=probabilities)

        hit = is_root(root, accessed_value)
        cost = get_depth(root, accessed_value)

        if cost is None:
            cost = 0

        root, _ = search(root, accessed_value)

        total_cost += cost
        total_hits += 1 if hit else 0
        value_access_counts[accessed_value] += 1
        value_hit_counts[accessed_value] += 1 if hit else 0
        value_cost_sums[accessed_value] += cost

    overall = OverallStats(
        total_accesses=num_accesses,
        total_hits=total_hits,
        total_cost=total_cost,
    )

    per_value = {
        val: ValueStats(
            value=val,
            access_count=value_access_counts[val],
            hit_count=value_hit_counts[val],
            total_cost=value_cost_sums[val],
        )
        for val in values
    }

    return overall, per_value


def print_results(
    overall: OverallStats,
    per_value: dict[int, ValueStats],
    probabilities: dict[int, float],
    num_top: int = 10,
) -> None:
    """Print formatted results table."""
    click.echo("\n=== Overall Statistics ===")
    click.echo(f"Total accesses: {overall.total_accesses:,}")
    click.echo(f"Total hits (already at root): {overall.total_hits:,}")
    click.echo(f"Hit rate: {overall.hit_rate:.2%}")
    click.echo(f"Average cost: {overall.avg_cost:.4f}")

    sorted_by_access = sorted(
        per_value.items(), key=lambda x: x[1].access_count, reverse=True
    )

    top_n = sorted_by_access[:num_top]
    bottom_n = sorted_by_access[-num_top:]

    click.echo(f"\n=== Top {num_top} Most Accessed Values ===")
    click.echo(
        f"{'Value':>6} {'Probability':>12} {'Accesses':>10} {'Hits':>8} {'Hit Rate':>10} {'Avg Cost':>12}"
    )
    click.echo("-" * 75)
    for val, stats in top_n:
        prob = probabilities[val]
        click.echo(
            f"{val:6d} {prob:12.6f} {stats.access_count:10,} {stats.hit_count:8,} "
            f"{stats.hit_rate:10.2%} {stats.avg_cost:12.4f}"
        )

    click.echo(f"\n=== Bottom {num_top} Least Accessed Values ===")
    click.echo(
        f"{'Value':>6} {'Probability':>12} {'Accesses':>10} {'Hits':>8} {'Hit Rate':>10} {'Avg Cost':>12}"
    )
    click.echo("-" * 75)
    for val, stats in reversed(bottom_n):
        prob = probabilities[val]
        click.echo(
            f"{val:6d} {prob:12.6f} {stats.access_count:10,} {stats.hit_count:8,} "
            f"{stats.hit_rate:10.2%} {stats.avg_cost:12.4f}"
        )

    top_avg_cost = sum(s.avg_cost for _, s in top_n) / len(top_n)
    bottom_avg_cost = sum(s.avg_cost for _, s in bottom_n) / len(bottom_n)

    click.echo("\n=== Comparison ===")
    click.echo(f"Average cost of top {num_top} most accessed: {top_avg_cost:.4f}")
    click.echo(
        f"Average cost of bottom {num_top} least accessed: {bottom_avg_cost:.4f}"
    )
    click.echo(
        f"Cost difference: {bottom_avg_cost - top_avg_cost:.4f} "
        f"({((bottom_avg_cost - top_avg_cost) / bottom_avg_cost * 100):.1f}% improvement)"
    )


@click.group()
def cli() -> None:
    """Splay tree access pattern analysis tools."""
    pass


@cli.command()
@click.option(
    "--num-values",
    default=50,
    help="Number of unique values in tree",
)
@click.option(
    "--num-accesses",
    default=100000,
    help="Number of random accesses to perform",
)
@click.option("--seed", default=42, help="Random seed for reproducibility")
@click.option(
    "--num-top",
    default=10,
    help="Number of top/bottom values to display",
)
def demo(num_values: int, num_accesses: int, seed: int, num_top: int) -> None:
    """
    Demonstrate splay tree access pattern behavior.

    Generates random access probabilities and performs many accesses,
    showing that frequently accessed values stay near the root.
    """
    click.echo(f"Generating {num_values} values with random access probabilities...")
    values = list(range(1, num_values + 1))
    probabilities_array = generate_probabilities(num_values, seed)
    probabilities_dict = {val: prob for val, prob in zip(values, probabilities_array)}

    click.echo(f"Running {num_accesses:,} random accesses...")
    overall, per_value = run_access_simulation(
        values, probabilities_array, num_accesses, seed
    )

    print_results(overall, per_value, probabilities_dict, num_top)


@cli.command()
def test() -> None:
    """Run the test suite."""
    pytest.main(["test_splay.py", "-v"])


if __name__ == "__main__":
    cli()
