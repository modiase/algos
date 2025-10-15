#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.pytest
from bisect import bisect_left
from collections.abc import Sequence

import pytest


class Solution:
    def lis(self, nums: Sequence[int]) -> int:
        if len(nums) == 0:
            return 0

        dp: list[int] = []
        for num in nums:
            left = bisect_left(dp, num)
            if left == len(dp):
                dp.append(num)
            else:
                dp[left] = min(dp[left], num)

        return len(dp)


@pytest.mark.parametrize(
    "nums, expected",
    [
        ([10, 9, 2, 5, 3, 7, 101, 18], 4),
        ([0, 1, 0, 3, 2, 3], 4),
        ([7, 7, 7, 7, 7, 7, 7], 1),
        ([], 0),
        ([1], 1),
        ([1, 2, 3, 4, 5], 5),
        ([5, 4, 3, 2, 1], 1),
    ],
)
def test_lis(nums: Sequence[int], expected: int):
    assert Solution().lis(nums) == expected


if __name__ == "__main__":
    pytest.main([__file__])
