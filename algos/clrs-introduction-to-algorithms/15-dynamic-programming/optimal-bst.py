#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.pytest -p python313Packages.loguru
"""
Scaffold for CLRS optimal binary search tree (OBST) exercises.

Defines a minimal `Node` structure, a `search` helper that reports comparison
counts, and an `optimal_bst` stub to be implemented later. The accompanying tests
provide probability distributions together with the optimal expected search cost,
so that once `optimal_bst` is implemented the tree it produces can be validated
by comparing its cost against the known optimum.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import pytest


@dataclass(slots=True, kw_only=True)
class Node:
    key: float
    left: Node | None = None
    right: Node | None = None


def search(root: Node | None, target: float) -> tuple[float | None, int]:
    """Return the located key (if any) and number of comparisons for `target`."""
    current = root
    cost = 0
    while current is not None:
        cost += 1
        if target == current.key:
            return current.key, cost
        if target < current.key:
            current = current.left
        else:
            current = current.right
    return None, cost + 1


def optimal_bst(
    keys: Sequence[float],
    key_probabilities: Sequence[float],
    dummy_key_probabilites: Sequence[float],
) -> Node | None:
    n = len(keys)
    if n == 0:
        return None

    if len(key_probabilities) != n:
        msg = "probabilities must match number of keys"
        raise ValueError(msg)
    if len(dummy_key_probabilites) != n + 1:
        msg = "dummy_probabilities must have length number_of_keys + 1"
        raise ValueError(msg)

    e = [[0.0 for _ in range(n + 1)] for _ in range(n + 2)]
    w = [[0.0 for _ in range(n + 1)] for _ in range(n + 2)]
    root = [[0 for _ in range(n + 1)] for _ in range(n + 2)]

    for i in range(1, n + 2):
        e[i][i - 1] = dummy_key_probabilites[i - 1]
        w[i][i - 1] = dummy_key_probabilites[i - 1]

    for length in range(1, n + 1):
        for i in range(1, n - length + 2):
            j = i + length - 1
            w[i][j] = w[i][j - 1] + key_probabilities[j - 1] + dummy_key_probabilites[j]
            best_cost = float("inf")
            best_root = i
            for r in range(i, j + 1):
                cost = e[i][r - 1] + e[r + 1][j] + w[i][j]
                if cost < best_cost:
                    best_cost = cost
                    best_root = r
            e[i][j] = best_cost
            root[i][j] = best_root

    def build(i: int, j: int) -> Node | None:
        if j < i:
            return None
        r = root[i][j]
        return Node(
            key=keys[r - 1],
            left=build(i, r - 1),
            right=build(r + 1, j),
        )

    return build(1, n)


def _dummy_targets(keys: Sequence[float]) -> list[float]:
    if not keys:
        return [0.0]
    targets: list[float] = [keys[0] - 0.5]
    for a, b in pairwise(keys):
        targets.append((a + b) / 2.0)
    targets.append(keys[-1] + 0.5)
    return targets


def _expected_cost(
    root: Node | None,
    keys: Sequence[float],
    probabilities: Sequence[float],
    dummy_probabilities: Sequence[float],
) -> float:
    total = 0.0
    for key, probability in zip(keys, probabilities):
        _, cost = search(root, key)
        total += probability * cost
    for target, probability in zip(_dummy_targets(keys), dummy_probabilities):
        _, cost = search(root, target)
        total += probability * cost
    return total


@dataclass(frozen=True, slots=True)
class OptimalBSTCase:
    keys: Sequence[float]
    probabilities: Sequence[float]
    dummy_probabilities: Sequence[float]
    expected_cost: float


@pytest.mark.parametrize(
    "case",
    (
        OptimalBSTCase(
            keys=(10, 20, 30),
            probabilities=(0.15, 0.10, 0.05),
            dummy_probabilities=(0.05, 0.10, 0.05, 0.05),
            expected_cost=1.25,
        ),
        OptimalBSTCase(
            keys=(10, 11),
            probabilities=(0.197586, 0.388449),
            dummy_probabilities=(0.087153, 0.171801, 0.155011),
            expected_cost=1.870505,
        ),
        OptimalBSTCase(
            keys=(10, 11, 12, 13, 14),
            probabilities=(0.142535, 0.097743, 0.021825, 0.084839, 0.016414),
            dummy_probabilities=(
                0.164827,
                0.167194,
                0.064617,
                0.060267,
                0.137169,
                0.04257,
            ),
            expected_cost=2.881762,
        ),
        OptimalBSTCase(
            keys=(10, 11),
            probabilities=(0.065842, 0.325312),
            dummy_probabilities=(0.144933, 0.158607, 0.305305),
            expected_cost=1.978229,
        ),
        OptimalBSTCase(
            keys=(10, 11, 12, 13),
            probabilities=(0.200623, 0.12072, 0.189666, 0.173088),
            dummy_probabilities=(0.041959, 0.139294, 0.034715, 0.011096, 0.088839),
            expected_cost=2.420967,
        ),
        OptimalBSTCase(
            keys=(10, 11, 12, 13, 14),
            probabilities=(0.034934, 0.035669, 0.012514, 0.117635, 0.167673),
            dummy_probabilities=(
                0.13961,
                0.159812,
                0.071935,
                0.110759,
                0.132618,
                0.01684,
            ),
            expected_cost=3.043504,
        ),
        OptimalBSTCase(
            keys=(10, 11),
            probabilities=(0.245176, 0.263598),
            dummy_probabilities=(0.178014, 0.108594, 0.204618),
            expected_cost=2.02301,
        ),
        OptimalBSTCase(
            keys=(10, 11, 12, 13, 14),
            probabilities=(0.120379, 0.09746, 0.1634, 0.172463, 0.074308),
            dummy_probabilities=(
                0.018414,
                0.024892,
                0.057556,
                0.079691,
                0.04767,
                0.143767,
            ),
            expected_cost=2.638019,
        ),
        OptimalBSTCase(
            keys=(10, 11, 12, 13, 14, 15),
            probabilities=(0.106178, 0.113428, 0.078248, 0.075458, 0.087843, 0.027378),
            dummy_probabilities=(
                0.100796,
                0.126532,
                0.02981,
                0.111612,
                0.10499,
                0.023742,
                0.013984,
            ),
            expected_cost=2.94075,
        ),
        OptimalBSTCase(
            keys=(10, 11, 12, 13, 14),
            probabilities=(0.126628, 0.126563, 0.077691, 0.036033, 0.115525),
            dummy_probabilities=(
                0.105104,
                0.055068,
                0.068384,
                0.10001,
                0.071065,
                0.117928,
            ),
            expected_cost=2.896992,
        ),
        OptimalBSTCase(
            keys=(10,),
            probabilities=(0.524204,),
            dummy_probabilities=(0.176398, 0.299398),
            expected_cost=1.475796,
        ),
        OptimalBSTCase(
            keys=(10,),
            probabilities=(0.239393,),
            dummy_probabilities=(0.405466, 0.355142),
            expected_cost=1.760607,
        ),
        OptimalBSTCase(
            keys=(10, 11, 12, 13, 14, 15),
            probabilities=(0.108033, 0.153418, 0.013846, 0.077155, 0.015482, 0.081617),
            dummy_probabilities=(
                0.125437,
                0.162892,
                0.153308,
                0.032323,
                0.013874,
                0.040775,
                0.02184,
            ),
            expected_cost=2.840228,
        ),
        OptimalBSTCase(
            keys=(10, 11, 12),
            probabilities=(0.032651, 0.278384, 0.126362),
            dummy_probabilities=(0.165082, 0.146519, 0.110049, 0.140954),
            expected_cost=2.284219,
        ),
        OptimalBSTCase(
            keys=(10, 11),
            probabilities=(0.173397, 0.239538),
            dummy_probabilities=(0.123555, 0.231502, 0.232008),
            expected_cost=2.115518,
        ),
        OptimalBSTCase(
            keys=(10,),
            probabilities=(0.368692,),
            dummy_probabilities=(0.243655, 0.387653),
            expected_cost=1.631308,
        ),
        OptimalBSTCase(
            keys=(10, 11, 12, 13),
            probabilities=(0.208927, 0.207287, 0.033693, 0.064542),
            dummy_probabilities=(0.02263, 0.233478, 0.012282, 0.196764, 0.020397),
            expected_cost=2.521004,
        ),
        OptimalBSTCase(
            keys=(10,),
            probabilities=(0.539271,),
            dummy_probabilities=(0.314822, 0.145908),
            expected_cost=1.460729,
        ),
        OptimalBSTCase(
            keys=(10,),
            probabilities=(0.422013,),
            dummy_probabilities=(0.161726, 0.41626),
            expected_cost=1.577987,
        ),
        OptimalBSTCase(
            keys=(10, 11, 12, 13, 14, 15),
            probabilities=(0.083458, 0.111465, 0.107482, 0.049679, 0.089314, 0.060997),
            dummy_probabilities=(
                0.096025,
                0.028907,
                0.126665,
                0.008988,
                0.082382,
                0.049661,
                0.104976,
            ),
            expected_cost=2.955198,
        ),
        OptimalBSTCase(
            keys=(10, 11),
            probabilities=(0.098476, 0.352065),
            dummy_probabilities=(0.345175, 0.029999, 0.174284),
            expected_cost=2.023109,
        ),
    ),
)
def test_expected_cost(case: OptimalBSTCase) -> None:
    assert _expected_cost(
        root=optimal_bst(
            keys=case.keys,
            key_probabilities=case.probabilities,
            dummy_key_probabilites=case.dummy_probabilities,
        ),
        keys=case.keys,
        probabilities=case.probabilities,
        dummy_probabilities=case.dummy_probabilities,
    ) == pytest.approx(case.expected_cost, rel=1e-5, abs=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, *sys.argv[1:]])
