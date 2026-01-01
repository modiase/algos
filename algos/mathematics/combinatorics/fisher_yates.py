#!/usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.pytest
"""
Fisher-Yates shuffle for generating uniform random permutations.

For each position i from n-1 down to 1:
    - Pick random j from [0, i]
    - Swap arr[i] with arr[j]

Time: O(n)
Space: O(1) - in-place
"""

from __future__ import annotations

import random
from collections import Counter

import pytest


def shuffle(arr: list, rng: random.Random | None = None) -> list:
    """Shuffle array in-place using Fisher-Yates algorithm."""
    if rng is None:
        rng = random.Random()

    for i in range(len(arr) - 1, 0, -1):
        j = rng.randint(0, i)
        arr[i], arr[j] = arr[j], arr[i]

    return arr


def test_produces_all_permutations() -> None:
    """All 24 permutations of [1,2,3,4] should eventually appear."""
    rng = random.Random(42)
    seen: set[tuple[int, ...]] = set()

    for _ in range(10000):
        arr = [1, 2, 3, 4]
        shuffle(arr, rng)
        seen.add(tuple(arr))
        if len(seen) == 24:
            break

    assert len(seen) == 24


def test_uniform_distribution() -> None:
    """All 24 permutations should appear roughly equally."""
    rng = random.Random(42)
    counts: Counter[tuple[int, ...]] = Counter()

    for _ in range(24000):
        arr = [1, 2, 3, 4]
        shuffle(arr, rng)
        counts[tuple(arr)] += 1

    expected = 1000
    for count in counts.values():
        assert abs(count - expected) < expected * 0.15


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
