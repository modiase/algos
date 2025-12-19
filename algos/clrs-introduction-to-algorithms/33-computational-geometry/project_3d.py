#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.click -p python313Packages.matplotlib
"""
Computes the 2D shadow of a 3D point cloud's convex hull via projection.

Correctness follows from the theorem: CH(proj(S)) = proj(CH(S)).

Projection is linear, hence preserves convex combinations. Therefore the
projection of a convex hull is convex, contains all projected points, and
is minimal. Computing the 2D hull of projected points yields the same
result as projecting the 3D hull.
"""

from __future__ import annotations

import math
import random
import subprocess
from dataclasses import dataclass

import click
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float

    def __sub__(self, other: Point2D) -> Point2D:
        return Point2D(self.x - other.x, self.y - other.y)


@dataclass(frozen=True)
class Point3D:
    x: float
    y: float
    z: float

    def project_xy(self) -> Point2D:
        """Project onto z=0 plane. Time: O(1)"""
        return Point2D(self.x, self.y)


def cross_product(o: Point2D, a: Point2D, b: Point2D) -> float:
    """Cross product for 2D Graham's scan. Time: O(1)"""
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)


def polar_angle(origin: Point2D, point: Point2D) -> float:
    """Compute polar angle from origin to point. Time: O(1)"""
    return math.atan2(point.y - origin.y, point.x - origin.x)


def grahams_scan(points: list[Point2D]) -> list[Point2D]:
    """
    Compute 2D convex hull using Graham's scan.

    Time: O(n log n)
    """
    if len(points) < 3:
        return list(points)

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

    return hull


@click.command()
@click.option("-n", "--num-points", default=20, help="Number of points to generate")
@click.option("--seed", default=42, help="Random seed")
@click.option("-o", "--output", default="/tmp/project_3d.png", help="Output file")
def demo(num_points: int, seed: int, output: str) -> None:
    """Visualize 3D point projection and convex hull shadow."""
    random.seed(seed)

    points_3d = [
        Point3D(
            random.uniform(0, 10),
            random.uniform(0, 10),
            random.uniform(0, 10),
        )
        for _ in range(num_points)
    ]

    points_2d = [p.project_xy() for p in points_3d]
    hull = grahams_scan(points_2d)

    fig = plt.figure(figsize=(16, 12))

    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    ax1.scatter(
        [p.x for p in points_3d],
        [p.y for p in points_3d],
        [p.z for p in points_3d],
        c="black",
        s=50,
    )
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title("3D Point Cloud")

    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    ax2.scatter(
        [p.x for p in points_3d],
        [p.y for p in points_3d],
        [p.z for p in points_3d],
        c="black",
        s=50,
    )
    for p in points_3d:
        ax2.plot([p.x, p.x], [p.y, p.y], [p.z, 0], "k--", linewidth=0.5, alpha=0.3)
    ax2.scatter(
        [p.x for p in points_2d],
        [p.y for p in points_2d],
        [0] * len(points_2d),
        c="gray",
        s=30,
        alpha=0.5,
    )
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.set_title("3D Points with Projections")

    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    ax3.scatter(
        [p.x for p in points_3d],
        [p.y for p in points_3d],
        [p.z for p in points_3d],
        c="black",
        s=50,
    )
    hull_x = [p.x for p in hull] + [hull[0].x]
    hull_y = [p.y for p in hull] + [hull[0].y]
    ax3.plot(hull_x, hull_y, [0] * len(hull_x), "r-", linewidth=2)
    ax3.add_collection3d(
        Poly3DCollection(
            [[(p.x, p.y, 0) for p in hull]],
            alpha=0.2,
            facecolor="red",
            edgecolor="none",
        )
    )
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Z")
    ax3.set_title("3D with Convex Hull Shadow")

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.scatter([p.x for p in points_2d], [p.y for p in points_2d], c="black", s=50)
    ax4.plot(hull_x, hull_y, "r-", linewidth=2)
    ax4.fill(hull_x, hull_y, "red", alpha=0.2)
    ax4.set_xlabel("X")
    ax4.set_ylabel("Y")
    ax4.set_aspect("equal")
    ax4.set_title("2D Convex Hull Shadow")
    ax4.grid(False)

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    click.echo(f"Convex hull shadow has {len(hull)} vertices")
    click.echo(f"Saved to: {output}")
    subprocess.run(f"open {output}", shell=True)


if __name__ == "__main__":
    demo()
