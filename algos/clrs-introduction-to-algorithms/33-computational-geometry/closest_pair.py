#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.pytest -p python313Packages.click -p python313Packages.matplotlib
"""
Closest pair of points using divide-and-conquer.

The algorithm finds the two closest points in a set of n points in O(n log n) time:

1. Presort points by x-coordinate and y-coordinate (O(n log n))

2. Divide: Split points into left and right halves by median x-coordinate

3. Conquer: Recursively find closest pairs in left and right halves
   - Base case: 3 or fewer points, use brute force

4. Combine: Check for closer pairs that span the dividing line
   - Only consider points within δ of the dividing line (δ = min of left/right distances)
   - For each point in the strip, only check next 7 points sorted by y-coordinate
   - This works because any 8 points in a δ × 2δ rectangle must have at least
     one pair closer than δ (pigeonhole principle)

The key insight is that we only need to check 7 points ahead in the y-sorted strip,
giving O(n) time for the combine step and O(n log n) overall.
"""

from __future__ import annotations

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

    def distance_to(self, other: Point2D) -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


def brute_force_closest(points: list[Point2D]) -> tuple[Point2D, Point2D, float]:
    """
    Find closest pair using brute force.

    Time: O(n²)
    """
    min_dist = float("inf")
    pair = (points[0], points[1])

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist = points[i].distance_to(points[j])
            if dist < min_dist:
                min_dist = dist
                pair = (points[i], points[j])

    return pair[0], pair[1], min_dist


def closest_pair_strip(
    strip: list[Point2D], delta: float
) -> tuple[Point2D, Point2D, float] | None:
    """
    Find closest pair in a vertical strip of width 2δ.

    Points must be sorted by y-coordinate. Only checks next 7 points for each point.

    Time: O(n) where n is the number of points in the strip
    """
    min_dist = delta
    pair = None

    for i in range(len(strip)):
        j = i + 1
        while j < len(strip) and (strip[j].y - strip[i].y) < min_dist:
            dist = strip[i].distance_to(strip[j])
            if dist < min_dist:
                min_dist = dist
                pair = (strip[i], strip[j])
            j += 1

    return (pair[0], pair[1], min_dist) if pair else None


def closest_pair_recursive(
    px: list[Point2D], py: list[Point2D]
) -> tuple[Point2D, Point2D, float]:
    """
    Find closest pair using divide-and-conquer.

    Args:
        px: Points sorted by x-coordinate
        py: Points sorted by y-coordinate (same points as px)

    Time: O(n log n)
    """
    n = len(px)

    if n <= 3:
        return brute_force_closest(px)

    mid = n // 2
    midpoint = px[mid]

    pyl = [p for p in py if p.x <= midpoint.x]
    pyr = [p for p in py if p.x > midpoint.x]

    p1_left, p2_left, delta_left = closest_pair_recursive(px[:mid], pyl)
    p1_right, p2_right, delta_right = closest_pair_recursive(px[mid:], pyr)

    if delta_left < delta_right:
        delta = delta_left
        best_pair = (p1_left, p2_left)
    else:
        delta = delta_right
        best_pair = (p1_right, p2_right)

    strip = [p for p in py if abs(p.x - midpoint.x) < delta]

    strip_result = closest_pair_strip(strip, delta)
    if strip_result and strip_result[2] < delta:
        return strip_result

    return best_pair[0], best_pair[1], delta


def closest_pair(points: list[Point2D]) -> tuple[Point2D, Point2D, float]:
    """
    Find the closest pair of points.

    Time: O(n log n)
    Space: O(n)
    """
    px = sorted(points, key=lambda p: p.x)
    py = sorted(points, key=lambda p: p.y)
    return closest_pair_recursive(px, py)


@pytest.mark.parametrize(
    "points, expected_distance",
    [
        (
            [Point2D(0, 0), Point2D(1, 1), Point2D(2, 2), Point2D(10, 10)],
            math.sqrt(2),
        ),
        (
            [Point2D(0, 0), Point2D(3, 0), Point2D(1, 1), Point2D(2, 1)],
            1.0,
        ),
        (
            [Point2D(0, 0), Point2D(1, 0), Point2D(0, 1)],
            1.0,
        ),
        (
            [
                Point2D(2, 3),
                Point2D(12, 30),
                Point2D(40, 50),
                Point2D(5, 1),
                Point2D(12, 10),
                Point2D(3, 4),
            ],
            math.sqrt(2),
        ),
    ],
)
def test_closest_pair(points: list[Point2D], expected_distance: float) -> None:
    _, _, dist = closest_pair(points)
    assert abs(dist - expected_distance) < 1e-9


def test_closest_pair_random() -> None:
    random.seed(42)
    points = [
        Point2D(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(50)
    ]

    _, _, divide_conquer_dist = closest_pair(points)
    _, _, brute_force_dist = brute_force_closest(points)

    assert abs(divide_conquer_dist - brute_force_dist) < 1e-9


@click.group()
def cli() -> None:
    """Closest pair of points tools."""
    pass


@cli.command()
@click.option("-n", "--num-points", default=30, help="Number of points to generate")
@click.option("--seed", default=42, help="Random seed")
@click.option("-o", "--output", default="/tmp/closest_pair.png", help="Output file")
def demo(num_points: int, seed: int, output: str) -> None:
    """Visualize closest pair of points."""
    random.seed(seed)

    points = [
        Point2D(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(num_points)
    ]

    p1, p2, dist = closest_pair(points)

    _, ax = plt.subplots(figsize=(10, 10))

    for p in points:
        ax.plot(p.x, p.y, "ko", markersize=6)

    ax.plot(
        [p1.x, p2.x],
        [p1.y, p2.y],
        "r-",
        linewidth=2,
        label=f"Closest pair (d={dist:.3f})",
    )
    ax.plot([p1.x, p2.x], [p1.y, p2.y], "ro", markersize=10)

    ax.annotate(
        f"({p1.x:.2f}, {p1.y:.2f})",
        (p1.x, p1.y),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "yellow", "alpha": 0.7},
    )
    ax.annotate(
        f"({p2.x:.2f}, {p2.y:.2f})",
        (p2.x, p2.y),
        xytext=(10, -20),
        textcoords="offset points",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "yellow", "alpha": 0.7},
    )

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(f"Closest Pair of Points (n={num_points})")

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    click.echo(f"Closest pair: ({p1.x:.2f}, {p1.y:.2f}) and ({p2.x:.2f}, {p2.y:.2f})")
    click.echo(f"Distance: {dist:.4f}")
    click.echo(f"Saved to: {output}")
    subprocess.run(f"open {output}", shell=True)


@cli.command()
def test() -> None:
    """Run the test suite."""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    cli()
