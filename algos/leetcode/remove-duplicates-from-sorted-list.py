"""
Remove duplicates from a sorted array in place such that the first
k numbers are a unique ordered sequence of non-decreasing numbers. 
The solution function should modify the array in place and return k.

Notes
=====
## Summary
T: 20,5
C: Y
PD: 2

## Comments

Need to be careful about the following things:
    * Array access after incrementing a variable.
    One must check again to ensure the variable is within bounds.
    * Not thinking about invariants carefully. After the while loop
    we know that a is pointing to the next number in the sequence
    provided that it has not reached the end (a == N) so we can swap
    without the superfluous check for count > 1 which I was previously
    doing.

tags: sorted-arrays, in-place
"""
from typing import List


def remove_duplicates(nums: List[int]) -> int:
    N = len(nums)
    a = 0
    b = 0
    while a < N:
        curr = nums[a]
        while a < N and nums[a] == curr:
            a += 1
        b += 1
        if a < N and b < N:  # EDIT: b <= N so if a < N then b < N necessarily
            nums[b] = nums[a]

    return b


def make_test_case(l, k):
    l0 = l[:]
    k0 = remove_duplicates(l0)
    assert k0 == k
    assert l0[:k0] == list(sorted(l0[:k0]))


def test_case_one():
    make_test_case([1, 2, 2, 3, 3, 4, 4, 5], 5)


def test_case_two():
    make_test_case([1, 2, 3], 3)


def test_case_three():
    make_test_case([1, 2, 2, 3], 3)


def test_case_four():
    make_test_case([1, 1, 1], 1)
