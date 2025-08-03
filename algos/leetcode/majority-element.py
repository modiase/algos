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

from typing import List


def majority_element(nums: List[int]) -> int:
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


def test_case_one():
    assert majority_element([1, 2, 1, 2, 1]) == 1


def test_case_two():
    assert majority_element([1, 2, 2, 2, 1]) == 2


def test_case_three():
    assert majority_element([1, 1, 2, 2, 1]) == 1


def test_case_four():
    assert majority_element([1, 1, 2, 2, 2]) == 2


def test_case_five():
    assert majority_element([1, 2, 3, 4, 2]) == 2


def test_case_six():
    assert majority_element([1, 2, 3, 5, 5]) == 5
