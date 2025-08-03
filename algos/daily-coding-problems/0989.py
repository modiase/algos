"""
You are given an array of length N, where each element i represents the number of
ways we can produce i units of change. For example, [1, 0, 1, 1, 2] would indicate
that there is only one way to make 0, 2, or 3 units, and two ways of making 4 unit
Given such an array, determine the denominations that must be in use. In the case
above, for example, there must be coins with value 2, 3, and 4.
"""

import itertools


def get_denominations(w):
    known = [list(itertools.dropwhile(lambda x: w[x] == 0, range(1, len(w))))[0]]
    possible = [i for i, c in enumerate(w, 0) if c >= 1 and i != 0 and i != known[0]]
    for p in possible:
        max_n = p // known[0]
        count = 0
        for i in range(1, max_n + 1):
            ways = itertools.combinations_with_replacement(known, i)
            for way in ways:
                if sum(way) == p:
                    count += 1
        if count == w[p] - 1:
            known += [p]
    return known


assert get_denominations([1, 0, 1, 1, 2]) == [2, 3, 4]
assert get_denominations([1, 0, 1, 1, 1]) == [2, 3]
assert get_denominations([1, 0, 1, 1, 1, 0, 2]) == [2, 3]
assert get_denominations([1, 0, 1, 1, 1, 0, 3]) == [2, 3, 6]
assert get_denominations([1, 0, 1, 1, 2, 0, 3]) == [2, 3, 4]
assert get_denominations([1, 0, 1, 1, 2, 0, 4]) == [2, 3, 4, 6]
