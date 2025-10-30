#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.pytest
"""
Given an array of integers nums containing n + 1 integers where each integer is in the range [1,
  n]
    inclusive.

    There is only one repeated number in nums, return this repeated number.

    You must solve the problem without modifying the array nums and using only constant extra space.



    Example 1:

    Input: nums = [1,3,4,2,2]
    Output: 2
    Example 2:

    Input: nums = [3,1,3,4,2]
    Output: 3
    Example 3:

    Input: nums = [3,3,3,3,3]
    Output: 3


    Constraints:

    1 <= n <= 105
    nums.length == n + 1
    1 <= nums[i] <= n
    All the integers in nums appear only once except for precisely one integer which appears two or more times.


    Follow up:

    How can we prove that at least one duplicate number must exist in nums?
    Can you solve the problem in linear runtime complexity?

"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from itertools import product

import pytest


"""
By the pigeon principle, we know that if there are n+1 numbers and they are
all in the range [1,n] then at least one of them is repeated.

The key insight here is that we can treat the numbers in the array as a
linked list. The linked list forms a graph where at least one vertex has in
degree two and all vertices have outdegree one. This guarantees a cycle. We
can then use Floyd's fast-pointer slow-pointer algo to find the in-degree
two node which is the start of the cycle.

Extra: if O(1) space is required but we relax the [1,n] constraint, we have
a problem where the optimal solution is now n lg R where R = max(a) - min(a).
This is shown in findDuplicateAlt.
"""


def findDuplicate(nums: Sequence[int]) -> int:
    fast = slow = nums[0]
    while True:
        fast = nums[nums[fast]]
        slow = nums[slow]
        if fast == slow:
            break

    fast = nums[0]
    while fast != slow:
        fast = nums[fast]
        slow = nums[slow]

    return fast


def findDuplicateAlt(nums: Sequence[int]) -> int:
    """
    Binary search the value range [min(nums), max(nums)] while counting how
    many elements fall on each side. Works in O(n log R) time without mutating
    nums. This works by the pigeonhole principle where we know that given there
    are R pigeonholes and R + 1 items there is at least one duplicated number.
    We use binary search to find the half with the duplicate and narrow down.
    """

    low, high = min(nums), max(nums)
    while low < high:
        mid = (low + high) // 2
        count = sum(low <= value <= mid for value in nums)
        expected = mid - low + 1
        if count > expected:
            high = mid
        else:
            low = mid + 1
    return low


TEST_CASES: Sequence[tuple[Sequence[int], int]] = (
    ([1, 3, 4, 2, 2], 2),
    ([3, 1, 3, 4, 2], 3),
    ([3, 3, 3, 3, 3], 3),
    ([1, 1], 1),
    ([1, 4, 6, 2, 5, 3, 6], 6),
    ([2, 5, 9, 6, 9, 3, 8, 9, 7, 1], 9),
    ([4, 3, 1, 4, 2], 4),
    ([5, 4, 3, 2, 1, 2], 2),
    ([2, 2, 2, 2], 2),
    ([6, 1, 2, 3, 4, 5, 6], 6),
    ([7, 7, 4, 3, 2, 1, 5, 6], 7),
    ([1, 2, 3, 4, 4, 5, 6, 7, 8], 4),
    ([8, 7, 1, 5, 4, 6, 2, 8, 3], 8),
    ([9, 1, 5, 3, 2, 4, 8, 7, 6, 9], 9),
    ([10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 10], 10),
    ([1, 5, 4, 3, 2, 6, 7, 8, 9, 5], 5),
    ([2, 1, 4, 3, 6, 5, 7, 8, 9, 10, 2], 2),
    ([11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 11], 11),
    ([12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 12], 12),
    ([4, 2, 1, 3, 4], 4),
)


@pytest.mark.parametrize(
    ("nums", "expected", "solution_fn"),
    [
        (nums, expected, solution_fn)
        for (nums, expected), solution_fn in product(
            TEST_CASES, (findDuplicate, findDuplicateAlt)
        )
    ],
)
def test_find_duplicate_variants(
    nums: Sequence[int], expected: int, solution_fn: Callable[[Sequence[int]], int]
):
    assert solution_fn(nums) == expected


if __name__ == "__main__":
    pytest.main([__file__, *sys.argv[1:]])
