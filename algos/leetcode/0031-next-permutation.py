"""A permutation of an array of integers is an arrangement of its members into
a sequence or linear order.

For example, for arr = [1,2,3], the following are all the permutations of arr:
[1,2,3], [1,3,2], [2, 1, 3], [2, 3, 1], [3,1,2], [3,2,1].  The next permutation
of an array of integers is the next lexicographically greater permutation of its
integer. More formally, if all the permutations of the array are sorted in one
container according to their lexicographical order, then the next permutation of
that array is the permutation that follows it in the sorted container. If such
arrangement is not possible, the array must be rearranged as the lowest possible
order (i.e., sorted in ascending order).

For example, the next permutation of arr = [1,2,3] is [1,3,2].  Similarly, the
next permutation of arr = [2,3,1] is [3,1,2].  While the next permutation of arr
= [3,2,1] is [1,2,3] because [3,2,1] does not have a lexicographical larger
rearrangement.  Given an array of integers nums, find the next permutation of
nums.

The replacement must be in place and use only constant extra memory.

Example 1:

Input: nums = [1,2,3] Output: [1,3,2]

Example 2:

Input: nums = [3,2,1] Output: [1,2,3]

Example 3:

Input: nums = [1,1,5] Output: [1,5,1]

"""

import sys
from collections.abc import Sequence
from itertools import islice

import pytest


class Solution:
    def nextPermutation(self, nums: list[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        N = len(nums)

        def pairwise(it, reverse=False):
            _it = lambda: reversed(it) if reverse else it  # noqa: E731
            return zip(_it(), islice(_it(), 1, None))

        def insort(idx):
            for i in range(idx, N):
                for j in range(i + 1, N):
                    if nums[i] > nums[j]:
                        nums[j], nums[i] = nums[i], nums[j]

        for i in range(N - 1, 0, -1):
            if nums[i - 1] < nums[i]:
                swap_idx = i
                for j in range(i + 1, N):
                    if nums[j] > nums[i - 1]:
                        swap_idx = j
                nums[swap_idx], nums[i - 1] = nums[i - 1], nums[swap_idx]
                insort(i)
                break
        else:
            insort(0)


@pytest.mark.parametrize(
    "nums, expected",
    [
        ([1, 2, 3], [1, 3, 2]),
        ([3, 2, 1], [1, 2, 3]),
        ([1, 1, 5], [1, 5, 1]),
        ([1, 3, 2], [2, 1, 3]),
        ([2, 1, 3], [2, 3, 1]),
        ([2, 3, 1], [3, 1, 2]),
        ([3, 1, 2], [3, 2, 1]),
        ([3, 2, 1], [1, 2, 3]),
        ([1, 5, 1], [5, 1, 1]),
        ([1, 5, 8, 4, 7, 6, 5, 3, 1], [1, 5, 8, 5, 1, 3, 4, 6, 7]),
    ],
)
def test_next_permutation(nums: Sequence[int], expected: Sequence[int]):
    s = Solution()
    s.nextPermutation(nums)
    assert nums == expected


if __name__ == "__main__":
    pytest.main([__file__, *sys.argv[1:]])
