#!/usr/bin/env nix-shell
#! nix-shell -i python3 -p python3 python3Packages.matplotlib python3Packages.numpy python3Packages.seaborn
"""
Heap Statistics Visualization Script

This script reads heap statistics from a JSON Lines file and creates histograms
to visualize the distribution of various statistics for skew and leftist heaps.
"""

import argparse
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def load_statistics(filename):
    """Load statistics from JSON Lines file."""
    skew_stats = defaultdict(list)
    leftist_stats = defaultdict(list)

    with open(filename, "r") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                heap_type = data["heap_type"]
                stats = data["statistics"]

                if heap_type == "skew":
                    for key, value in stats.items():
                        skew_stats[key].append(value)
                elif heap_type == "leftist":
                    for key, value in stats.items():
                        leftist_stats[key].append(value)
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse line: {e}")
                continue

    return skew_stats, leftist_stats


def create_histograms(skew_stats, leftist_stats, output_prefix="heap_analysis"):
    """Create histograms for all statistics."""
    # Set up the plotting style
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (15, 10)

    # Get all statistic keys
    all_keys = set(skew_stats.keys()) | set(leftist_stats.keys())

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Heap Statistics Distribution Analysis", fontsize=16, fontweight="bold"
    )

    # Flatten axes for easier iteration
    axes = axes.flatten()

    for idx, stat_key in enumerate(sorted(all_keys)):
        if idx >= len(axes):
            break

        ax = axes[idx]

        # Get data for both heap types
        skew_data = skew_stats.get(stat_key, [])
        leftist_data = leftist_stats.get(stat_key, [])

        if not skew_data and not leftist_data:
            continue

        # Create histogram
        if skew_data:
            ax.hist(
                skew_data,
                bins=30,
                alpha=0.7,
                label="Skew Heap",
                color="skyblue",
                edgecolor="black",
                density=True,
            )
        if leftist_data:
            ax.hist(
                leftist_data,
                bins=30,
                alpha=0.7,
                label="Leftist Heap",
                color="lightcoral",
                edgecolor="black",
                density=True,
            )

        # Add labels and title
        ax.set_xlabel(stat_key.replace("_", " ").title())
        ax.set_ylabel("Density")
        ax.set_title(f"{stat_key.replace('_', ' ').title()} Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add summary statistics
        if skew_data:
            skew_mean = np.mean(skew_data)
            skew_std = np.std(skew_data)
            ax.axvline(
                skew_mean,
                color="blue",
                linestyle="--",
                label=f"Skew Mean: {skew_mean:.2f}",
            )

        if leftist_data:
            leftist_mean = np.mean(leftist_data)
            leftist_std = np.std(leftist_data)
            ax.axvline(
                leftist_mean,
                color="red",
                linestyle="--",
                label=f"Leftist Mean: {leftist_mean:.2f}",
            )

    # Hide unused subplots
    for idx in range(len(all_keys), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_histograms.png", dpi=300, bbox_inches="tight")
    plt.show()

    return fig


def create_comparison_plots(skew_stats, leftist_stats, output_prefix="heap_analysis"):
    """Create comparison plots for key statistics."""
    sns.set_style("whitegrid")

    # Create box plots for comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Heap Statistics Comparison", fontsize=16, fontweight="bold")

    # Prepare data for box plots
    stat_keys = ["size", "min_depth", "max_depth"]

    for idx, stat_key in enumerate(stat_keys):
        if idx >= len(axes):
            break

        ax = axes[idx // 2, idx % 2]

        # Prepare data for seaborn boxplot
        data = []
        labels = []

        if stat_key in skew_stats:
            data.extend(skew_stats[stat_key])
            labels.extend(["Skew"] * len(skew_stats[stat_key]))

        if stat_key in leftist_stats:
            data.extend(leftist_stats[stat_key])
            labels.extend(["Leftist"] * len(leftist_stats[stat_key]))

        if data:
            # Create box plot
            sns.boxplot(data=data, ax=ax)
            ax.set_ylabel(stat_key.replace("_", " ").title())
            ax.set_title(f"{stat_key.replace('_', ' ').title()} Comparison")

            # Add individual data points
            if stat_key in skew_stats:
                ax.scatter(
                    [0] * len(skew_stats[stat_key]),
                    skew_stats[stat_key],
                    alpha=0.5,
                    color="skyblue",
                    s=20,
                )
            if stat_key in leftist_stats:
                ax.scatter(
                    [1] * len(leftist_stats[stat_key]),
                    leftist_stats[stat_key],
                    alpha=0.5,
                    color="lightcoral",
                    s=20,
                )

            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Skew", "Leftist"])

    # Hide unused subplots
    for idx in range(len(stat_keys), len(axes.flatten())):
        axes.flatten()[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

    return fig


def print_summary_statistics(skew_stats, leftist_stats):
    """Print summary statistics for both heap types."""
    print("=" * 60)
    print("HEAP STATISTICS SUMMARY")
    print("=" * 60)

    all_keys = set(skew_stats.keys()) | set(leftist_stats.keys())

    for stat_key in sorted(all_keys):
        print(f"\n{stat_key.replace('_', ' ').upper()}:")
        print("-" * 40)

        if stat_key in skew_stats:
            skew_data = skew_stats[stat_key]
            print(f"Skew Heap (n={len(skew_data)}):")
            print(f"  Mean: {np.mean(skew_data):.2f}")
            print(f"  Std:  {np.std(skew_data):.2f}")
            print(f"  Min:  {np.min(skew_data):.2f}")
            print(f"  Max:  {np.max(skew_data):.2f}")
            print(f"  Q25:  {np.percentile(skew_data, 25):.2f}")
            print(f"  Q50:  {np.percentile(skew_data, 50):.2f}")
            print(f"  Q75:  {np.percentile(skew_data, 75):.2f}")

        if stat_key in leftist_stats:
            leftist_data = leftist_stats[stat_key]
            print(f"Leftist Heap (n={len(leftist_data)}):")
            print(f"  Mean: {np.mean(leftist_data):.2f}")
            print(f"  Std:  {np.std(leftist_data):.2f}")
            print(f"  Min:  {np.min(leftist_data):.2f}")
            print(f"  Max:  {np.max(leftist_data):.2f}")
            print(f"  Q25:  {np.percentile(leftist_data, 25):.2f}")
            print(f"  Q50:  {np.percentile(leftist_data, 50):.2f}")
            print(f"  Q75:  {np.percentile(leftist_data, 75):.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and visualize heap statistics"
    )
    parser.add_argument("input_file", help="Input JSON Lines file with heap statistics")
    parser.add_argument(
        "--output",
        "-o",
        default="heap_analysis",
        help="Output prefix for generated plots (default: heap_analysis)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots, only print summary statistics",
    )

    args = parser.parse_args()

    try:
        print(f"Loading statistics from {args.input_file}...")
        skew_stats, leftist_stats = load_statistics(args.input_file)

        if not skew_stats and not leftist_stats:
            print("No statistics found in the file!")
            return

        print(f"Loaded {sum(len(v) for v in skew_stats.values())} skew heap records")
        print(
            f"Loaded {sum(len(v) for v in leftist_stats.values())} leftist heap records"
        )

        # Print summary statistics
        print_summary_statistics(skew_stats, leftist_stats)

        if not args.no_plots:
            print("\nGenerating plots...")

            # Create histograms
            print("Creating histograms...")
            create_histograms(skew_stats, leftist_stats, args.output)

            # Create comparison plots
            print("Creating comparison plots...")
            create_comparison_plots(skew_stats, leftist_stats, args.output)

            print("\nPlots saved as:")
            print(f"  {args.output}_histograms.png")
            print(f"  {args.output}_comparison.png")

    except FileNotFoundError:
        print(f"Error: File '{args.input_file}' not found!")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
