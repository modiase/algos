"""
Use the following ideas to develop a nonrecursive, linear time algorithm
for the maximum-subarray problem. Start at the left of the array, and
progress toward the right, keeping track of the maximum subarray seen
so far. Knowing a maximum subarray of A[1...j], extend the answer to
find a maximum subarray of A[i...j+1], for some 1 <= i <= j + 1. 
Determine a maximum subarray of the form A[i...j+1] in constant time
based on knowing the maximum subarray ending at index j.
"""
import importlib
import math as m
import random as rn
import time
from typing import List, Tuple

ex414 = importlib.import_module(
    package="introduction-to-algorithms", name='4-1-4')
quadratic_max_subarray = ex414.max_subarray
brute_force_max_subarray = ex414.brute_force_max_subarray


def linear_max_subarray(a: List[int]) -> Tuple[int, int, int]:
    sum = a[0]
    l = 0
    r = 0
    n = len(a)

    rolling_sum = sum
    for i in range(1, n):
        rolling_sum += a[i]
        if rolling_sum > sum:
            r = i
            sum = rolling_sum
        if a[i] > sum:
            l = r = i
            sum = rolling_sum = a[i]

    return (sum, l, r)


def test_one():
    assert linear_max_subarray([-2, 1, 3, -1, 5]) == (8, 1, 4)


def test_two():
    assert linear_max_subarray([-2, 1, 3, 1, -5]) == (5, 1, 3)


def test_three():
    assert linear_max_subarray([-2, -1, -3, -1, 5]) == (5, 4, 4)


def test_four():
    assert linear_max_subarray([7, -1, -3, -1, 6]) == (8, 0, 4)


def test_five():
    assert linear_max_subarray([4, -1, -3, -1, 1, 1, 6]) == (8, 4, 6)


if __name__ == '__main__':
    rn.seed(12345)
    powers = [x / 4 for x in range(1, 25)]
    test_arrays = [[rn.randint(-127, 127)
                    for _ in range(0, int(pow(10, p)))] for p in powers]

    brutes = []
    quads = []
    lins = []
    for input in test_arrays:
        result = []
        brute = None
        if len(input) <= 10000:
            brute = []
            for i in range(0, 3):
                start = time.perf_counter()
                brute_force_max_subarray(input)
                end = time.perf_counter()
                brute.append(end-start)
            brute = sum(brute) / 3
            brutes.append(m.log(brute, 10))

        quad = []
        for i in range(0, 3):
            start = time.perf_counter()
            quadratic_max_subarray(input)
            end = time.perf_counter()
            quad.append(end-start)
        quad = sum(quad) / 3
        quads.append(m.log(quad, 10))

        lin = []
        for i in range(0, 3):
            start = time.perf_counter()
            linear_max_subarray(input)
            end = time.perf_counter()
            lin.append(end-start)
        lin = sum(lin) / 3
        lins.append(m.log(lin, 10))

        if brute is not None:
            print(
                f'N: {len(input)} , brute: {brute:.3E}s , quad: {quad:.3E}s, linear: {lin:.3E}s')
        else:
            print(
                f'N: {len(input)} , brute: X s , quad: {quad:.3E}s, linear: {lin:.3E}s')

    print(f'Brute force slope: {(brutes[15] - brutes[0]) / 2}')
    print(f'Quadratic slope: {(quads[23] - quads[0])/ 3}')
    print(f'Linear slope: {(lins[23] - lins[0])/ 3}')
