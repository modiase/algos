#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.pytest
from __future__ import annotations

from collections.abc import Sequence

import pytest


def subsequence_sum_2d(nums: Sequence[int], target: int) -> bool:
    """
    Time: O(len(nums) * target)
    """
    n = len(nums)
    dp = [[False] * (target + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = True

    for i in range(1, n + 1):
        for j in range(target + 1):
            dp[i][j] = dp[i - 1][j]

            if nums[i - 1] <= j:
                dp[i][j] = dp[i][j] or dp[i - 1][j - nums[i - 1]]

    return dp[n][target]


def subsequence_sum_optimized(nums: Sequence[int], target: int) -> bool:
    """
    Space optimised. Uses space: O(target) as well as O(len(nums) * target) time.
    The sum does not care which row the sum is in, only that it was previously computed.
    """
    dp = [False] * (target + 1)
    dp[0] = True

    for num in nums:
        for j in range(target, num - 1, -1):
            dp[j] = dp[j] or dp[j - num]

    return dp[target]


@pytest.mark.parametrize(
    "nums, target, expected",
    [
        ([], 0, True),
        ([], 5, False),
        ([1], 1, True),
        ([1], 2, False),
        ([1, 2, 3], 4, True),
        ([1, 2, 3], 6, True),
        ([1, 2, 3], 7, False),
        ([2, 3, 7], 10, True),
        ([2, 3, 7], 11, False),
        ([1, 5, 11, 5], 11, True),
        ([1, 5, 11, 5], 10, True),
        ([1, 5, 11, 5], 15, False),
    ],
)
def test_subsequence_sum_2d(nums: Sequence[int], target: int, expected: bool) -> None:
    assert subsequence_sum_2d(nums, target) == expected


@pytest.mark.parametrize(
    "nums, target, expected",
    [
        ([], 0, True),
        ([], 5, False),
        ([1], 1, True),
        ([1], 2, False),
        ([1, 2, 3], 4, True),
        ([1, 2, 3], 6, True),
        ([1, 2, 3], 7, False),
        ([2, 3, 7], 10, True),
        ([2, 3, 7], 11, False),
        ([1, 5, 11, 5], 11, True),
        ([1, 5, 11, 5], 10, True),
        ([1, 5, 11, 5], 15, False),
    ],
)
def test_subsequence_sum_optimized(
    nums: Sequence[int], target: int, expected: bool
) -> None:
    assert subsequence_sum_optimized(nums, target) == expected


def test_both_implementations_agree() -> None:
    """Verify both implementations produce the same results."""
    test_cases = [
        ([1, 2, 3, 4, 5], 9),
        ([1, 2, 3, 4, 5], 15),
        ([1, 2, 3, 4, 5], 16),
        ([10, 20, 30], 40),
        ([10, 20, 30], 50),
        ([10, 20, 30], 61),
    ]

    for nums, target in test_cases:
        result_2d = subsequence_sum_2d(nums, target)
        result_opt = subsequence_sum_optimized(nums, target)
        assert result_2d == result_opt, f"Mismatch for {nums}, {target}"


if __name__ == "__main__":
    pytest.main([__file__])
