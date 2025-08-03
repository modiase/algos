"""
Write a function that rotates a list by k elements. For example, [1, 2, 3, 4, 5, 6]
rotated by two becomes [3, 4, 5, 6, 1, 2]. Try solving this without creating a copy
of the list. How many swap or move operations do you need?
"""

from typing import List


def swap(xs: List[int], i1: int, i2: int) -> None:
    if xs[i1] == xs[i2]:
        return

    xs[i1] += xs[i2]
    xs[i2] = -(xs[i2] - xs[i1])
    xs[i1] -= xs[i2]


def solution(xs: List[int], k: int) -> List[int]:
    if k == 0:
        return xs

    n = len(xs)
    l = n // k

    for a in range(0, l):
        for i in range(0, k):
            j = n - a * k - i - 1
            swap(xs, k - i - 1, j)

    return xs


assert solution([1, 2, 3, 4, 5], 2) == [3, 4, 5, 1, 2]
assert solution([1, 2, 3, 4, 5, 6], 2) == [3, 4, 5, 6, 1, 2]
assert solution([1, 2, 3, 4, 5, 6], 3) == [4, 5, 6, 1, 2, 3]
assert solution([1, 2, 3, 4, 5, 6], 4) == [5, 6, 1, 2, 3, 4]
assert solution([1, 2, 3, 4, 5, 6], 5) == [6, 1, 2, 3, 4, 5]
assert solution([1, 2], 1) == [2, 1]
assert solution([1], 0) == [1]
assert solution([], 0) == []
