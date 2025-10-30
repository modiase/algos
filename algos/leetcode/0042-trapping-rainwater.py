#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.pytest
"""
# Trapping Rain Water - Solution Progression

## Core Principle
Water at position i is trapped up to: **min(max_left, max_right) - height[i]**

---

## Approach 1: BFS Draining from Edges
**Thinking:** Water escapes from edges. Use BFS to see which cells can drain at each level.

**Insight discovered:** The edges are the only escape routes. Water level is determined by the shortest boundary to reach an edge.

**Complexity:** O(nm) time/space where m = max height

---

## Approach 2: Two Pointers from Edges
**Thinking:** If edges control everything, just track the two boundary heights.

**Key realization:** We don't need to know BOTH boundaries exactly - only which one is smaller.
- If left_max < right_max: the right side is "tall enough", so left_max limits position left
- If right_max < left_max: the left side is "tall enough", so right_max limits position right

**Process:** Start at both edges, maintain max heights seen, move the pointer with smaller boundary inward.

**Complexity:** O(n) time, O(1) space

---

## Bridge: BFS → Two Pointers

1. **BFS shows:** Edges are the critical boundaries
2. **Simplify:** For 1D, only two edges exist - track their max heights
3. **Optimize:** Instead of precomputing all maxes, maintain them as you process from both ends simultaneously
4. **Final insight:** Always process the side with the smaller boundary - it's the limiting factor

---

## Remember
- Water escapes over the shorter wall
- We process from edges inward because edges give us certainty about boundaries
- Moving the smaller boundary guarantees the other side is "protective enough"
- Water is ultimately trapped in some 'valley'. We can use two-pointers to
  construct sub problems to find at each index the relevant 'valley'.
"""

from __future__ import annotations

import sys
from collections import deque
from collections.abc import Sequence
from itertools import product

import pytest


def trap_bfs(height: list[int]) -> int:
    """
    0 - empty
    1 - black
    2 - water
    """
    if not height:
        return 0
    N = len(height) + 2
    m = max(height)
    if m == 0:
        return 0

    def neighbours(row, col):
        if row - 1 >= 0:
            yield row - 1, col
        if row + 1 < N:
            yield row + 1, col
        if col + 1 < m:
            yield row, col + 1
        if row - 1 >= 0 and col + 1 < m:
            yield row - 1, col + 1
        if row + 1 < N and col + 1 < m:
            yield row + 1, col + 1

    queue = deque([*[(0, i) for i in range(m)], *[(N - 1, i) for i in range(m)]])
    grid = [[0] * m]
    grid.extend([[*[1] * h, *[2] * (m - h)] for h in height])
    grid.append([0] * m)

    while queue:
        cell = queue.popleft()
        for row, col in neighbours(*cell):
            if grid[row][col] == 2:
                queue.append((row, col))
                grid[row][col] = 0

    return sum(1 for row in range(N) for col in range(m) if grid[row][col] == 2)


def trap(height: Sequence[int]) -> int:
    left, right = 0, len(height) - 1
    max_left, max_right = 0, 0
    trapped_water = 0

    while left < right:
        if height[left] < height[right]:
            if height[left] >= max_left:
                max_left = height[left]
            else:
                trapped_water += max_left - height[left]
            left += 1
        else:
            if height[right] >= max_right:
                max_right = height[right]
            else:
                trapped_water += max_right - height[right]
            right -= 1

    return trapped_water


TEST_CASES: Sequence[tuple[Sequence[int], int]] = (
    ((), 0),
    ([0], 0),
    ([0, 0], 0),
    ([1, 0, 1], 1),
    ([4, 2, 0, 3, 2, 5], 9),
    ([2, 0, 2], 2),
    ([3, 0, 0, 2, 0, 4], 10),
    ([1, 2, 3, 4, 5], 0),
    ([5, 4, 3, 2, 1], 0),
    ([1, 3, 2, 1, 2, 1, 5, 1, 2, 1], 7),
    ([2, 1, 0, 1, 3], 4),
    ([0, 3, 0], 0),
    ([3, 1, 2, 1, 2, 1, 3], 8),
    ([2, 0, 2, 0, 2], 4),
    ([2, 1, 2], 1),
    ([3, 2, 1, 3], 3),
    ([1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1], 6),
    ([0, 2, 0, 1, 0, 3, 0], 5),
    ([9, 6, 8, 8, 5, 6, 3], 3),
    ([5, 2, 1, 2, 1, 5], 14),
)


@pytest.mark.parametrize(
    ("height", "expected", "function"),
    [
        (height, expected, function)
        for (height, expected), function in product(TEST_CASES, (trap, trap_bfs))
    ],
)
def test_trap_variants(
    height: Sequence[int],
    expected: int,
    function,
) -> None:
    assert function(height) == expected


if __name__ == "__main__":
    pytest.main([__file__, *sys.argv[1:]])
