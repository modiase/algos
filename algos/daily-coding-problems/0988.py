"""
Given a positive integer n, find the smallest number of squared integers which sum to n.

For example, given n = 13, return 2 since 13 = 3^2 + 2^2 = 9 + 4.

Given n = 27, return 3 since 27 = 3^2 + 3^2 + 3^2 = 9 + 9 + 9.
"""

import math
import itertools


def smallest_rec(n):
    squares = [x * x for x in range(1, math.ceil(math.sqrt(n)))]

    def _all_possible(acc, remaining):
        result = []
        if remaining < 0:
            return []
        if remaining == 0:
            return [acc]
        for square in squares:
            result += _all_possible([*acc, square], remaining - square)
        return result

    valid = _all_possible([], n)
    return min([len(t) for t in valid])


def smallest_it(n):
    squares = [x * x for x in range(1, math.ceil(math.sqrt(n)))]
    for i in range(0, n):
        ts = itertools.product(squares, repeat=i)
        for t in ts:
            if sum(t) == n:
                return len(t)


assert smallest_rec(13) == 2
assert smallest_rec(27) == 3
assert smallest_it(13) == 2
assert smallest_it(27) == 3
