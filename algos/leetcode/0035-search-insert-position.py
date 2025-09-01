import sys

import pytest


class Solution:
    def searchInsert(self, nums: list[int], target: int) -> int:
        if len(nums) == 0:
            return 0
        lo, hi = 0, len(nums) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if target <= nums[mid]:
                hi = mid
            else:
                lo = mid + 1

        while lo < len(nums) and nums[lo] < target:
            lo += 1
        return lo


@pytest.mark.parametrize(
    "nums, target, expected",
    [
        ([1, 3, 5, 6], 5, 2),
        ([1, 3, 5, 6], 2, 1),
        ([1, 3, 5, 6], 7, 4),
        ([1, 3, 5, 6], 0, 0),
        ([1, 3, 5, 6], 4, 2),
        ([1], 1, 0),
        ([1], 2, 1),
        ([1], 0, 0),
        ([], 5, 0),
        ([2, 2, 2, 2], 2, 0),
        ([1, 1, 1], 1, 0),
        ([0, 0, 0, 0, 0], 0, 0),
        ([1, 2, 3, 4, 5], 1, 0),
        ([1, 2, 3, 4, 5], 5, 4),
        ([1, 2, 2, 3], 1, 0),
        ([1, 2, 2, 3], 3, 3),
        ([1, 2, 3, 4, 5], 6, 5),
        ([1, 2, 3, 4, 5], 0, 0),
        ([2, 4, 6, 8], 3, 1),
        ([2, 4, 6, 8], 7, 3),
        ([1, 1, 2, 2, 2, 3], 1, 0),
        ([1, 1, 2, 2, 2, 3], 2, 2),
        ([1, 1, 2, 2, 2, 3], 3, 5),
        ([1, 1, 1, 1], 1, 0),
        ([1000, 2000, 2000, 3000], 2000, 1),
        ([1000, 2000, 2000, 3000], 1000, 0),
        ([1000, 2000, 2000, 3000], 3000, 3),
        ([-5, -3, -3, -1, 0], -3, 1),
        ([-5, -3, -3, -1, 0], -5, 0),
        ([-5, -3, -3, -1, 0], 0, 4),
        ([-5, -3, -3, -1, 0], -2, 3),
        ([-2, -1, 0, 1, 1, 2], 1, 3),
        ([-2, -1, 0, 1, 1, 2], -1, 1),
        ([-2, -1, 0, 1, 1, 2], 0, 2),
        ([1, 1, 2, 3, 4], 1, 0),
        ([1, 2, 3, 4, 4], 4, 3),
        ([1, 1, 2, 2, 3, 3], 2, 2),
    ],
)
def test_solution(nums: list[int], target: int, expected: int):
    assert Solution().searchInsert(nums, target) == expected


if __name__ == "__main__":
    pytest.main([__file__, *sys.argv[1:]])
