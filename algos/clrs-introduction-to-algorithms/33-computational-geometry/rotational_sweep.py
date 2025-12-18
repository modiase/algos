#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.click -p python313Packages.matplotlib
"""
Rotational sweep visualization with polar angle computation.

Generates random points and computes their polar angles relative to the centroid
using a pole (reference direction) pointing rightward along the positive x-axis
from the centroid.
"""

from __future__ import annotations

import math
import random
import subprocess
from dataclasses import dataclass

import click
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float

    def __sub__(self, other: Point2D) -> Point2D:
        return Point2D(self.x - other.x, self.y - other.y)


def polar_angle(centroid: Point2D, point: Point2D) -> float:
    """
    Compute polar angle of point relative to centroid.

    Uses pole direction (1, 0) as reference (positive x-axis) and computes
    angle using cross product and dot product with atan2. Returns angle in
    [0, 2π) measured counter-clockwise from the pole.

    Time: O(1)
    """
    relative = point - centroid
    reference = Point2D(1, 0)

    angle = math.atan2(
        reference.x * relative.y - reference.y * relative.x,
        reference.x * relative.x + reference.y * relative.y,
    )
    return angle if angle >= 0 else angle + 2 * math.pi


@click.command()
@click.option("-n", "--num-points", default=20, help="Number of points to generate")
@click.option("--seed", default=42, help="Random seed")
@click.option("-o", "--output", default="/tmp/rotational_sweep.png", help="Output file")
def main(num_points: int, seed: int, output: str) -> None:
    """Visualize rotational sweep with polar angles."""
    random.seed(seed)

    points = [
        Point2D(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(num_points)
    ]

    centroid = Point2D(
        sum(p.x for p in points) / len(points),
        sum(p.y for p in points) / len(points),
    )

    max_distance = max(
        math.sqrt((p.x - centroid.x) ** 2 + (p.y - centroid.y) ** 2) for p in points
    )
    pole = Point2D(centroid.x + max_distance, centroid.y)

    points_with_angles = [(p, polar_angle(centroid, p)) for p in points]
    points_with_angles.sort(key=lambda x: x[1])
    point_ranks = {p: rank + 1 for rank, (p, _) in enumerate(points_with_angles)}

    _, ax = plt.subplots(figsize=(12, 12))

    ax.plot(centroid.x, centroid.y, "o", color="gray", markersize=10)
    ax.annotate(
        "centroid",
        (centroid.x, centroid.y),
        xytext=(-15, -15),
        textcoords="offset points",
        fontsize=10,
        color="gray",
    )

    ax.plot(
        [centroid.x, pole.x],
        [centroid.y, centroid.y],
        "k--",
        linewidth=1,
        alpha=0.3,
    )

    for p in points:
        ax.plot([centroid.x, p.x], [centroid.y, p.y], "k-", linewidth=0.5, alpha=0.4)
        ax.plot(p.x, p.y, "ko", markersize=6)

        angle = polar_angle(centroid, p)
        rank = point_ranks[p]
        rel = p - centroid
        label = f"#{rank} ({rel.x:.1f},{rel.y:.1f})\n∠{angle:.2f}"
        ax.annotate(
            label, (p.x, p.y), xytext=(5, 5), textcoords="offset points", fontsize=7
        )

    ax.set_aspect("equal")
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    click.echo(f"Saved to: {output}")
    subprocess.run(f"open {output}", shell=True)


if __name__ == "__main__":
    main()
