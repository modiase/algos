"""
Remove duplicates from a sorted array in place such that the first
k numbers are an ordered sequence of non-decreasing numbers where
each number in the sequence may appear at most twice. The
solution function should modify the array in place and return k.

Notes
=====
## Summary
T: 15
C: Y
PD: 1

## Comments

This was fairly straightforward after doing the first exercise
since I was able to leverage that solution. Need to be careful
about ensuring you know the state of a variable at each line
- one issue I ran into was solved by ensuring I assigned a
value before incrementing rather than after.

Furthermore, I realised that there was an invariant where
b <= a always such that if a < N then necessarily b < N
so I can eliminate that duplicate check from the solution.
Comment added to the previous solution.

tags: sorted-arrays, in-place
"""

from typing import List


def remove_duplicates(nums: List[int]) -> int:
    N = len(nums)
    a = 0
    b = 0
    while a < N:
        curr = nums[a]
        count = 0
        while a < N and nums[a] == curr:
            a += 1
            count += 1
            if a < N and count == 2:
                nums[b] = curr
                b += 1
        b += 1
        if a < N and b < N:
            nums[b] = nums[a]

    return b


def make_test_case(arr, k):
    l0 = arr[:]
    k0 = remove_duplicates(l0)
    assert k0 == k
    assert l0[:k0] == list(sorted(l0[:k0]))


def test_case_one():
    make_test_case([1, 2, 2, 3, 3, 4, 4, 5], 8)


def test_case_two():
    make_test_case([1, 2, 3], 3)


def test_case_three():
    make_test_case([1, 2, 2, 3], 4)


def test_case_four():
    make_test_case([1, 1, 1], 2)
