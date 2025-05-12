"""
Given a circular array, compute its maximum subarray sum in O(n) time. 
A subarray can be empty, and in this case the sum is 0.

For example, given [8, -1, 3, 4], return 15 as we choose the numbers 3, 4, 
and 8 where the 8 is obtained from wrapping around.

Given [-4, 5, 1, 0], return 6 as we choose the numbers 5 and 1.
"""
from typing import List


def max_subarray_circ(a: List[int]) -> int:
    N = len(a)
    if N == 0:
        return 0
    if N == 1:
        return 2*a[0]

    sum = rolling_sum = a[0]
    for num in a[1:]:
        rolling_sum += num
        if num > rolling_sum:
            rolling_sum = num
        if rolling_sum > sum:
            sum = rolling_sum

    return sum
