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
T: 90
PD: 5

## Comments

I don't know why I found this so hard. Even when I looked up the answer
I still struggled to get an implementation. I think eventually I slowed 
down and looked properly at what I was trying to do, but it was a comedy
of errors until that point.

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
"""

from typing import List

def merge_sorted_list(nums1: List[int], n: int, nums2: List[int], m: int):
    if n == 0:
        j = m - 1
        while j > - 1:
            nums1[j] = nums2[j]

    p2 = m + n - 1
    i = n - 1
    j = m - 1

    while j >= 0 and i >= 0 and p2 >= 0:
        if nums1[i] >  nums2[j]:
            nums1[p2] = nums1[i]
            i -= 1
        else:
            nums1[p2] = nums2[j]
            j -= 1
        p2 -= 1

    while j >= 0:
        print(p2)
        nums1[p2] = nums2[j]
        p2 -= 1
        j -= 1






a0 = [1,2,3,0,0,0]
b0 = [2,5,6]
merge_sorted_list(a0, len(a0)-len(b0), b0, len(b0))
assert a0 == [1,2,2,3,5,6]

a1 = [2,5,6,0,0,0]
b1 = [1,2,3]
merge_sorted_list(a1, len(a1)-len(b1), b1, len(b1))
assert a1 == [1,2,2,3,5,6]


a2 = [1,1,1,0,0,0]
b2 = [7,8,9]
merge_sorted_list(a2, len(a2)-len(b2), b2, len(b2))
assert a2 == [1,1,1,7,8,9]

a3 = [8,13,0]
b3 = [1]
merge_sorted_list(a3, len(a3)-len(b3), b3, len(b3))
assert a3 == [1,8,13]

a4 = [8,13]
b4 = []
merge_sorted_list(a4, len(a4)-len(b4), b4, len(b4))
assert a4 == [8,13]


