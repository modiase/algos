"""
Merge Sorted Array
You are given two integer arrays nums1 and nums2, sorted in non-decreasing orde
, and two integers m and n, representing the number of elements in nums1 and nums2 respectively.

Merge nums1 and nums2 into a single array sorted in non-decreasing order.

The final sorted array should not be returned by the function, but instead be stored
inside the array nums1. To accommodate this, nums1 has a length of m + n, where
the first m elements denote the elements that should be merged, and the last n elements
are set to 0 and should be ignored. nums2 has a length of n.


# Notes

## Review
C: N
T: 90,30
PD: 5

## Comments

I don't know why I found this so hard. Even when I looked up the answer
I still struggled to get an implementation. 

I think eventually I slowed 
down and looked properly at what I was trying to do, but it was a comedy
of errors until that point.

EDIT: I then had to put in some more time to fix my solution. I made
it more complicated than necessary by not paying attention to my limits.
Eventually, I simplified it down. This is one I should return to because 
it is fundamental and really shouldn't have taken this long.

## Workings

[1,2,3**,0,0,0*]
[2,5,6***]

[1,2,3**,0,0*,6]
[2,5***,6]

[1,2,3**,0*,5,6]
[2***,5,6]

[1,2**,3*,3,5,6]
[2***,5,6]

[1,2** *,2,3,5,6]
***[2,5,6]

*[1,2,2,3,5,6]
*[2,5,6]

tags: limits, arrays, sorted-arrays, off-by-one
"""

from typing import List


def merge_sorted_list(nums1: List[int], m: int, nums2: List[int], n: int):
    p2 = n + m - 1
    i = m - 1
    j = n - 1

    while j >= 0:
        if i >= 0 and nums1[i] > nums2[j]:
            nums1[p2] = nums1[i]
            i -= 1
        else:
            nums1[p2] = nums2[j]
            j -= 1
        p2 -= 1


def make_test_case(a, b):
    expected = sorted(a[:len(a)-len(b)] + b)
    acpy = a[:]
    bcpy = b[:]
    merge_sorted_list(a, len(a)-len(b), b, len(b))
    print(f'a: {acpy}, m: {len(a)-len(b)}, b: {bcpy}, n: {len(b)}, expected: {expected}, result: {a}')
    assert a == expected


def test_case_one():
    a = [1, 2, 3, 0, 0, 0]
    b = [2, 5, 6]
    make_test_case(a, b)


def test_case_two():
    a = [2, 5, 6, 0, 0, 0]
    b = [1, 2, 3]
    make_test_case(a, b)


def test_case_three():
    a = [1, 1, 1, 0, 0, 0]
    b = [7, 8, 9]
    make_test_case(a, b)


def test_case_four():
    a = [8, 13, 0]
    b = [1]
    make_test_case(a, b)


def test_case_five():
    a = [8, 13]
    b = []
    make_test_case(a, b)


def test_case_six():
    a = [0, 0]
    b = [8, 13]
    make_test_case(a, b)
