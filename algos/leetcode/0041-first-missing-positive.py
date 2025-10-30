#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.pytest
"""
Given an unsorted integer array nums. Return the smallest positive integer that is not present in nums.

You must implement an algorithm that runs in O(n) time and uses O(1) auxiliary space.



Example 1:

Input: nums = [1,2,0]
Output: 3
Explanation: The numbers in the range [1,2] are all in the array.
Example 2:

Input: nums = [3,4,-1,1]
Output: 2
Explanation: 1 is in the array but 2 is missing.
Example 3:

Input: nums = [7,8,9,11,12]
Output: 1
Explanation: The smallest positive integer 1 is missing.
"""

from __future__ import annotations

import sys
from collections.abc import MutableSequence

import pytest


class Solution:
    """
    The key thing to realise here is that the setup of the problem gives every
    advantage. We can only use O(1) auxiliary space which means we can modify
    the array. By the pigeonhole principle, the array either contains positive
    integers [1,n] in which case the first missive positive is n + 1 otherwise
    we can map each element to its correct location in the array in linear time
    and then scan the array to find the first index which is *not* a fixed
    point.
    """

    def firstMissingPositive(self, nums: MutableSequence[int]) -> int:
        size, start = len(nums), 0
        while start < size:
            current = nums[start]
            idx = current - 1
            while size > idx >= 0 and nums[idx] != current:
                nums[idx], current = current, nums[idx]
                idx = current - 1
            start += 1

        for candidate in range(1, size + 1):
            if nums[candidate - 1] != candidate:
                return candidate

        return max(nums) + 1 if nums else 1


@pytest.mark.parametrize(
    ("nums", "expected"),
    [
        ([1, 2, 0], 3),
        ([3, 4, -1, 1], 2),
        ([7, 8, 9, 11, 12], 1),
        ([1, 1], 2),
        ([2], 1),
        ([1, 2, 3], 4),
        ([2, 3, 4, 5, 6], 1),
        ([1], 2),
        ([1, 2, 2, 1, 3, 1, 0, 4, 0], 5),
    ],
)
def test_first_missing_positive(nums: MutableSequence[int], expected: int):
    actual = Solution().firstMissingPositive(list(nums))
    assert actual == expected


if __name__ == "__main__":
    pytest.main([__file__, *sys.argv[1:]])
