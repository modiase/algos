#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.click -p python313Packages.matplotlib -p python313Packages.pytest
"""
Graham's scan algorithm for computing 2D convex hull.

The convex hull is the smallest convex polygon containing all points.
Graham's scan computes it in O(n log n) time by:
1. Finding the lowest point (starting point)
2. Sorting all other points by polar angle relative to start
3. Building the hull by maintaining a stack and checking for left turns

A left turn (counter-clockwise) means the point is on the hull boundary.
A right turn (clockwise) means we need to backtrack and remove the previous point.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
import math
import random
import subprocess
from dataclasses import dataclass

import click
import matplotlib.pyplot as plt
import pytest


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float

    def __sub__(self, other: Point2D) -> Point2D:
        return Point2D(self.x - other.x, self.y - other.y)


def cross_product(o: Point2D, a: Point2D, b: Point2D) -> float:
    """
    Cross product of vectors (a-o) and (b-o).

    Returns positive for counter-clockwise turn (left turn),
    negative for clockwise turn (right turn), zero for collinear.

    Time: O(1)
    """
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)


def polar_angle(origin: Point2D, point: Point2D) -> float:
    """Compute polar angle from origin to point. Time: O(1)"""
    return math.atan2(point.y - origin.y, point.x - origin.x)


def grahams_scan(points: Collection[Point2D]) -> Sequence[Point2D]:
    """
    Compute convex hull using Graham's scan.

    Time: O(n log n)
    """
    if len(points) < 3:
        return tuple(points)

    start = min(points, key=lambda p: (p.y, p.x))

    sorted_points = sorted(
        [p for p in points if p != start],
        key=lambda p: (
            polar_angle(start, p),
            (p.x - start.x) ** 2 + (p.y - start.y) ** 2,
        ),
    )

    hull = [start, sorted_points[0]]

    for point in sorted_points[1:]:
        while len(hull) > 1 and cross_product(hull[-2], hull[-1], point) <= 0:
            hull.pop()
        hull.append(point)

    return tuple(hull)


@pytest.mark.parametrize(
    "points, expected_size",
    [
        ([Point2D(0, 0), Point2D(1, 0), Point2D(0, 1)], 3),
        ([Point2D(0, 0), Point2D(2, 0), Point2D(1, 0), Point2D(1, 1)], 3),
        (
            [
                Point2D(0, 0),
                Point2D(3, 0),
                Point2D(3, 3),
                Point2D(0, 3),
                Point2D(1, 1),
            ],
            4,
        ),
    ],
)
def test_grahams_scan(points: list[Point2D], expected_size: int) -> None:
    assert len(grahams_scan(points)) == expected_size


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option("-n", "--num-points", default=20, help="Number of points to generate")
@click.option("--seed", default=42, help="Random seed")
@click.option("-o", "--output", default="/tmp/convex_hull.png", help="Output file")
def demo(num_points: int, seed: int, output: str) -> None:
    random.seed(seed)

    points = [
        Point2D(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(num_points)
    ]

    hull = grahams_scan(points)

    _, ax = plt.subplots(figsize=(10, 10))

    for p in points:
        ax.plot(p.x, p.y, "ko", markersize=6)

    hull_x = [p.x for p in hull] + [hull[0].x]
    hull_y = [p.y for p in hull] + [hull[0].y]
    ax.plot(hull_x, hull_y, "r-", linewidth=2, alpha=0.7)
    ax.fill(hull_x, hull_y, "red", alpha=0.1)

    for i, p in enumerate(hull):
        ax.plot(p.x, p.y, "ro", markersize=8)
        ax.annotate(
            f"#{i + 1}",
            (p.x, p.y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            color="red",
        )

    ax.set_aspect("equal")
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    click.echo(f"Convex hull has {len(hull)} vertices")
    click.echo(f"Saved to: {output}")
    subprocess.run(f"open {output}", shell=True)


@cli.command()
def test() -> None:
    """Run the test suite."""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    cli()
