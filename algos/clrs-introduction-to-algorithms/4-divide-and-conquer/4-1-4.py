"""
Implement both the brute-force and recursive algorithms for the maximum
subarray problem on your own computer. What problem size n_0 gives the
crossover point at which the recursive algorithm beats the brute-force
algorithm? Then, change the base case of the recursive algorithm to use
the brute-force algorithm whenever the problem size is below n_0. Does
that change the crossover point?
"""

import math as m
import random as rn
import time
from typing import List, Tuple


def find_max_of_length(a: List[int], n: int, i: int) -> Tuple[int, int, int]:
    sum = 0
    for j in range(0, i):
        sum += a[j]

    msum = sum
    l = 0
    r = i - 1

    for j in range(0, n - i):
        sum = sum - a[j] + a[j + i]
        if sum > msum:
            msum = sum
            l = j + 1
            r = j + i
    return (msum, l, r)


def brute_force_max_subarray(a: List[int]) -> Tuple[int, int, int]:
    n = len(a)
    msum = float("-inf")
    ml = -1
    mr = -1

    for i in range(0, n + 1):
        (sum, l, r) = find_max_of_length(a, n, i)
        if sum > msum:
            ml = l
            mr = r
            msum = sum
    return (int(msum), ml, mr)


def _max_crossing_subarray(
    a: List[int], n: int, low: int, mid: int, high: int
) -> Tuple[int, int, int]:
    if low == high:
        return (a[low], low, high)
    l = r = mid
    _l = mid - 1
    _r = mid + 1
    msum = a[mid]

    lsum = a[mid]
    if _l >= 0:
        while _l >= low:
            lsum += a[_l]
            if lsum > msum:
                l = _l
                msum = lsum
            _l -= 1
    else:
        l = 0

    rsum = msum
    if _r < n:
        while _r <= high:
            rsum += a[_r]
            if rsum > msum:
                r = _r
                msum = rsum
            _r += 1
    else:
        r = n - 1

    return (msum, l, r)


def max_subarray(a: List[int]) -> Tuple[int, int, int]:
    n = len(a)

    def _max_subarray(a: List[int], low: int, high: int) -> Tuple[int, int, int]:
        if low == high:
            return (a[low], low, high)
        mid = (low + high) // 2
        results = []
        results.append(_max_subarray(a, low, mid))
        results.append(_max_subarray(a, min(mid + 1, high), high))
        results.append(_max_crossing_subarray(a, n, low, mid, high))
        return sorted(results, key=lambda x: -x[0])[0]

    return _max_subarray(a, 0, n - 1)


def test_brute_force_one():
    assert brute_force_max_subarray([-2, 1, 3, -1, 5]) == (8, 1, 4)


def test_brute_force_two():
    assert brute_force_max_subarray([-2, 1, 3, 1, -5]) == (5, 1, 3)


def test_brute_force_three():
    assert brute_force_max_subarray([-2, -1, -3, -1, 5]) == (5, 4, 4)


def test_brute_force_four():
    assert brute_force_max_subarray([7, -1, -3, -1, 6]) == (8, 0, 4)


def test_crossing_one():
    assert _max_crossing_subarray([-2, 1, 3, -1, 5], 5, 2, 2, 2) == (3, 2, 2)


def test_crossing_two():
    assert _max_crossing_subarray([-2, 1, 3, -1, 5], 5, 1, 2, 2) == (4, 1, 2)


def test_crossing_three():
    assert _max_crossing_subarray([-2, 1, 3, -1, 5], 5, 1, 2, 4) == (8, 1, 4)


def test_crossing_four():
    assert _max_crossing_subarray([-2, 1, 3, -1, 5], 5, 0, 0, 1) == (-1, 0, 1)


def test_crossing_five():
    assert _max_crossing_subarray([-2, 1, 3, -1, 5], 5, 0, 1, 1) == (1, 1, 1)


def test_crossing_six():
    assert _max_crossing_subarray([2, -1, 3, -1, 5], 5, 0, 2, 4) == (8, 0, 4)


def test_one():
    assert max_subarray([-2, 1, 3, -1, 5]) == (8, 1, 4)


def test_two():
    assert max_subarray([-2, 1, 3, 1, -5]) == (5, 1, 3)


def test_three():
    assert max_subarray([-2, -1, -3, -1, 5]) == (5, 4, 4)


def test_four():
    assert max_subarray([7, -1, -3, -1, 6]) == (8, 0, 4)


if __name__ == "__main__":
    rn.seed(12345)
    powers = [x / 4 for x in range(1, 17)]
    test_arrays = [
        [rn.randint(-127, 127) for _ in range(0, int(pow(10, p)))] for p in powers
    ]

    brutes = []
    quads = []
    for input in test_arrays:
        result = []
        brute = []
        for i in range(0, 3):
            start = time.perf_counter()
            brute_force_max_subarray(input)
            end = time.perf_counter()
            brute.append(end - start)
        brute = sum(brute) / 3
        brutes.append(m.log(brute, 10))

        quad = []
        for i in range(0, 3):
            start = time.perf_counter()
            max_subarray(input)
            end = time.perf_counter()
            quad.append(end - start)
        quad = sum(quad) / 3
        quads.append(m.log(quad, 10))

        print(f"N: {len(input)} , brute: {brute:.3E}s , quad: {quad:.3E}s")

    print(f"Brute force slope: {(brutes[15] - brutes[0]) / 2}")
    print(f"Quadratic slope: {(quads[15] - quads[0]) / 2}")
