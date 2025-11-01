#!/usr/bin/env python
"""
Given an array, find the majority element which is defined
as the element for which there are at least n/2 occurences.
Can you do it in O(n) time and O(1) additional space?


Notes
=====
## Summary
T: 5
C: Y (for simple implmentation, had to look up Boye-Moore Majority Voting Algorithm)
PD: 5

tags: in-place, arrays, statistics
"""

import unittest

from collections.abc import Sequence


def majority_element(nums: Sequence[int]) -> int:
    """
    Find the majority element in O(len(num)) time using Boyer-Moore majority
    vote.
    """
    N = len(nums)
    count = 1
    i = nums[0]
    idx = 1
    while idx < N:
        count += -1 + 2 * int(nums[idx] == i)
        if count < 0:
            i = nums[idx]
            count = 1
        idx += 1

    # N.B., if there is no guarantee that a majority
    # element exists we would have to loop through
    # again to guarantee that the candidate is in-
    # fact the majority element. The overall algorithm
    # is still O(n) time.

    return i


class TestMajorityElement(unittest.TestCase):
    def test_case_one(self):
        assert majority_element([1, 2, 1, 2, 1]) == 1

    def test_case_two(self):
        assert majority_element([1, 2, 2, 2, 1]) == 2

    def test_case_three(self):
        assert majority_element([1, 1, 2, 2, 1]) == 1

    def test_case_four(self):
        assert majority_element([1, 1, 2, 2, 2]) == 2

    def test_case_five(self):
        assert majority_element([1, 2, 3, 4, 2]) == 2

    def test_case_six(self):
        assert majority_element([1, 2, 3, 5, 5]) == 5


if __name__ == "__main__":
    unittest.main()
