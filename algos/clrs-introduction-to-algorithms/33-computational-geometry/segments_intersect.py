#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.pytest -p python313Packages.click -p python313Packages.matplotlib
"""
Line segment intersection detection using cross products.

The algorithm uses cross products to detect intersection in O(1) time:

1. Direction test: For each segment, compute which side of its line the
   other segment's endpoints fall on using cross products:
   - Cross product > 0 → clockwise turn (point is to the right)
   - Cross product < 0 → counterclockwise turn (point is to the left)
   - Cross product = 0 → collinear

2. Straddle check: Segments intersect if endpoints straddle each other.
   When two direction values have opposite signs (one positive, one negative),
   it means the segment crosses the infinite line defined by the other segment.
   This is called "straddling."

   One segment straddling is insufficient for intersection.
   Segment p1-p2 could cross the infinite line through p3-p4, but the actual
   segment p3-p4 might not reach that crossing point. Therefore we need BOTH:
   - p1 and p2 are on opposite sides of line p3-p4 (d1 and d2 have opposite signs)
   - AND p3 and p4 are on opposite sides of line p1-p2 (d3 and d4 have opposite signs)

   When both segments straddle each other's infinite lines, they must intersect
   at the unique point where those lines cross.

3. Collinear cases: If any endpoint is collinear (d = 0), check if it lies
   on the actual segment between the segment's endpoints
"""

from __future__ import annotations

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


def direction(pi: Point2D, pj: Point2D, pk: Point2D) -> float:
    """
    Compute direction of point pk relative to directed line segment pi→pj.

    Returns cross product (pk - pi) × (pj - pi):
    - Positive: pk is to the right (clockwise turn)
    - Negative: pk is to the left (counterclockwise turn)
    - Zero: pk is collinear with pi-pj

    Time: O(1)
    """
    return (pk.x - pi.x) * (pj.y - pi.y) - (pk.y - pi.y) * (pj.x - pi.x)


def on_segment(pi: Point2D, pj: Point2D, pk: Point2D) -> bool:
    """
    Check if point pk lies on segment pi-pj (assumes pk is collinear).

    Time: O(1)
    """
    return min(pi.x, pj.x) <= pk.x <= max(pi.x, pj.x) and min(
        pi.y, pj.y
    ) <= pk.y <= max(pi.y, pj.y)


def segments_intersect(p1: Point2D, p2: Point2D, p3: Point2D, p4: Point2D) -> bool:
    """
    Determine if line segments p1-p2 and p3-p4 intersect.

    Time: O(1)
    """
    d1 = direction(p3, p4, p1)
    d2 = direction(p3, p4, p2)
    d3 = direction(p1, p2, p3)
    d4 = direction(p1, p2, p4)

    return (d1 * d2 < 0 and d3 * d4 < 0) or any(
        d == 0 and on_segment(pi, pj, pk)
        for d, pi, pj, pk in [
            (d1, p3, p4, p1),
            (d2, p3, p4, p2),
            (d3, p1, p2, p3),
            (d4, p1, p2, p4),
        ]
    )


@pytest.mark.parametrize(
    "p1, p2, p3, p4, expected",
    [
        (Point2D(0, 0), Point2D(2, 2), Point2D(0, 2), Point2D(2, 0), True),
        (Point2D(0, 0), Point2D(4, 0), Point2D(2, -1), Point2D(2, 1), True),
        (Point2D(0, 0), Point2D(1, 1), Point2D(2, 0), Point2D(3, 1), False),
        (Point2D(0, 0), Point2D(1, 0), Point2D(2, 0), Point2D(3, 0), False),
        (Point2D(0, 0), Point2D(1, 1), Point2D(1, 1), Point2D(2, 0), True),
        (Point2D(0, 0), Point2D(2, 0), Point2D(1, 0), Point2D(3, 0), True),
        (Point2D(0, 0), Point2D(4, 4), Point2D(1, 1), Point2D(3, 3), True),
    ],
)
def test_segments_intersect(
    p1: Point2D, p2: Point2D, p3: Point2D, p4: Point2D, expected: bool
) -> None:
    assert segments_intersect(p1, p2, p3, p4) == expected


def test_direction_values() -> None:
    assert direction(Point2D(0, 0), Point2D(1, 0), Point2D(1, 1)) < 0
    assert direction(Point2D(0, 0), Point2D(1, 0), Point2D(1, -1)) > 0
    assert direction(Point2D(0, 0), Point2D(1, 0), Point2D(2, 0)) == 0


@click.group()
def cli() -> None:
    """Line segment intersection tools."""
    pass


@cli.command()
@click.option("--p1", nargs=2, type=float, help="First point of segment 1")
@click.option("--p2", nargs=2, type=float, help="Second point of segment 1")
@click.option("--p3", nargs=2, type=float, help="First point of segment 2")
@click.option("--p4", nargs=2, type=float, help="Second point of segment 2")
@click.option("--seed", type=int, default=42, help="Random seed for generation")
@click.option("-o", "--output", default="/tmp/segments.png", help="Output file")
def demo(
    p1: tuple[float, float] | None,
    p2: tuple[float, float] | None,
    p3: tuple[float, float] | None,
    p4: tuple[float, float] | None,
    seed: int,
    output: str,
) -> None:
    """Visualize segment intersection with direction annotations."""
    random.seed(seed)

    if p1 is None:
        p1 = (random.uniform(0, 4), random.uniform(0, 4))
    if p2 is None:
        p2 = (random.uniform(0, 4), random.uniform(0, 4))
    if p3 is None:
        p3 = (random.uniform(0, 4), random.uniform(0, 4))
    if p4 is None:
        p4 = (random.uniform(0, 4), random.uniform(0, 4))

    pt1, pt2 = Point2D(*p1), Point2D(*p2)
    pt3, pt4 = Point2D(*p3), Point2D(*p4)

    intersects = segments_intersect(pt1, pt2, pt3, pt4)

    _, ax = plt.subplots(figsize=(10, 10))

    ax.plot([pt1.x, pt2.x], [pt1.y, pt2.y], color="#AEC6CF", linewidth=2)
    ax.plot([pt3.x, pt4.x], [pt3.y, pt4.y], color="#FFB6C1", linewidth=2)

    points = [("p1", pt1), ("p2", pt2), ("p3", pt3), ("p4", pt4)]
    for name, pt in points:
        ax.plot(pt.x, pt.y, "ko", markersize=8)
        ax.annotate(
            name,
            (pt.x, pt.y),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=12,
            fontweight="bold",
        )

    d1 = direction(pt3, pt4, pt1)
    d2 = direction(pt3, pt4, pt2)
    d3 = direction(pt1, pt2, pt3)
    d4 = direction(pt1, pt2, pt4)

    def draw_direction_arrow(
        pi: Point2D, pk: Point2D, d: float, color: str, label: str
    ) -> None:
        vec = Point2D(pk.x - pi.x, pk.y - pi.y)
        if (length := (vec.x**2 + vec.y**2) ** 0.5) > 0:
            vec = Point2D(vec.x / length * 0.5, vec.y / length * 0.5)
        ax.arrow(
            pi.x,
            pi.y,
            vec.x,
            vec.y,
            head_width=0.15,
            head_length=0.1,
            fc=color,
            ec=color,
            alpha=0.6,
        )
        ax.text(
            pi.x + vec.x,
            pi.y + vec.y,
            f"{label}={d:.1f}\n{'CCW' if d > 0 else 'CW' if d < 0 else 'COLIN'}",
            fontsize=9,
            ha="center",
            bbox={"boxstyle": "round", "facecolor": color, "alpha": 0.3},
        )

    draw_direction_arrow(pt3, pt1, d1, "#B0E0A8", "d1")
    draw_direction_arrow(pt3, pt2, d2, "#98D8C8", "d2")
    draw_direction_arrow(pt1, pt3, d3, "#FFDAB9", "d3")
    draw_direction_arrow(pt1, pt4, d4, "#E6B8E6", "d4")

    ax.set_aspect("equal")
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    click.echo(f"Segments intersect: {intersects}")
    click.echo(f"Visualization saved to: {output}")
    click.echo(f"Direction values: d1={d1:.2f}, d2={d2:.2f}, d3={d3:.2f}, d4={d4:.2f}")
    subprocess.run(f"open {output}", shell=True)


@cli.command()
def test() -> None:
    """Run the test suite."""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    cli()
