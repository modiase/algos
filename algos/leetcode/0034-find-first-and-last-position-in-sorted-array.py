import sys

import pytest


class Solution:
    @staticmethod
    def bisect_left(nums: list[int], target: int) -> int:
        N = len(nums)
        if N == 0:
            return -1
        lo, hi = 0, N - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if target <= nums[mid]:
                hi = mid
            else:
                lo = mid + 1

        return lo if nums[lo] == target else -1

    @staticmethod
    def bisect_right(nums: list[int], target: int) -> int:
        N = len(nums)
        if N == 0:
            return -1
        lo, hi = 0, N - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if target < nums[mid]:
                hi = mid - 1
            else:
                lo = mid

        return hi if nums[hi] == target else -1

    def searchRange(self, nums: list[int], target: int) -> list[int]:
        return [self.bisect_left(nums, target), self.bisect_right(nums, target)]


@pytest.mark.parametrize(
    "nums, target, expected",
    [
        ([5, 7, 7, 8, 8, 10], 8, [3, 4]),
        ([5, 7, 7, 8, 8, 10], 6, [-1, -1]),
        ([1], 1, [0, 0]),
        ([1], 2, [-1, -1]),
        ([0], 0, [0, 0]),
        ([], 5, [-1, -1]),
        ([2, 2, 2, 2], 2, [0, 3]),
        ([1, 1, 1], 1, [0, 2]),
        ([0, 0, 0, 0, 0], 0, [0, 4]),
        ([1, 2, 3, 4, 5], 1, [0, 0]),
        ([1, 2, 3, 4, 5], 5, [4, 4]),
        ([1, 2, 2, 3], 1, [0, 0]),
        ([1, 2, 2, 3], 3, [3, 3]),
        ([1, 2, 3, 4, 5], 6, [-1, -1]),
        ([1, 2, 3, 4, 5], 0, [-1, -1]),
        ([2, 4, 6, 8], 3, [-1, -1]),
        ([2, 4, 6, 8], 7, [-1, -1]),
        ([1, 1, 2, 2, 2, 3], 1, [0, 1]),
        ([1, 1, 2, 2, 2, 3], 2, [2, 4]),
        ([1, 1, 2, 2, 2, 3], 3, [5, 5]),
        ([1, 1, 1, 1], 1, [0, 3]),
        ([1000, 2000, 2000, 3000], 2000, [1, 2]),
        ([1000, 2000, 2000, 3000], 1000, [0, 0]),
        ([1000, 2000, 2000, 3000], 3000, [3, 3]),
        ([-5, -3, -3, -1, 0], -3, [1, 2]),
        ([-5, -3, -3, -1, 0], -5, [0, 0]),
        ([-5, -3, -3, -1, 0], 0, [4, 4]),
        ([-5, -3, -3, -1, 0], -2, [-1, -1]),
        ([-2, -1, 0, 1, 1, 2], 1, [3, 4]),
        ([-2, -1, 0, 1, 1, 2], -1, [1, 1]),
        ([-2, -1, 0, 1, 1, 2], 0, [2, 2]),
        ([1, 1, 2, 3, 4], 1, [0, 1]),
        ([1, 2, 3, 4, 4], 4, [3, 4]),
        ([1, 1, 2, 2, 3, 3], 2, [2, 3]),
    ],
)
def test_solution(nums: list[int], target: int, expected: list[int]):
    assert Solution().searchRange(nums, target) == expected


if __name__ == "__main__":
    pytest.main([__file__, *sys.argv[1:]])
