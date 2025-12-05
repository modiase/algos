#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.pytest
from __future__ import annotations

from collections.abc import Sequence

import pytest


def subarray_sum(nums: Sequence[int], target: int) -> list[tuple[int, int]]:
    seen_sums: dict[int, int] = {0: -1}
    prefix_sum = 0
    result = []

    for idx, n in enumerate(nums):
        prefix_sum += n

        if (diff := prefix_sum - target) in seen_sums:
            result.append((seen_sums[diff] + 1, idx))

        seen_sums[prefix_sum] = idx

    return result


@pytest.mark.parametrize(
    "nums, target, expected",
    [
        ([], 3, []),
        ([3], 3, [(0, 0)]),
        ([3, 3], 3, [(0, 0), (1, 1)]),
        ([1, 2, 3, 4], 3, [(0, 1), (2, 2)]),
        ([1, 1, 1], 2, [(0, 1), (1, 2)]),
        ([1, 2, 3, 6], 6, [(0, 2), (3, 3)]),
        ([5, -2, 3, 1], 4, [(2, 3)]),
        ([1, -1, 1, -1], 0, [(0, 1), (1, 2), (2, 3)]),
        ([2, 3, 1, 4], 5, [(0, 1), (2, 3)]),
        ([1, 2, 1, 2, 1], 3, [(0, 1), (1, 2), (2, 3), (3, 4)]),
    ],
)
def test_subarray_sum(
    nums: Sequence[int], target: int, expected: list[tuple[int, int]]
) -> None:
    assert subarray_sum(nums, target) == expected


if __name__ == "__main__":
    pytest.main([__file__])
