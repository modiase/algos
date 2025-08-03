"""
Given a list of elements, find the majority element, which appears more than half the time (> floor(len(lst) / 2.0)).

You can assume that such element exists.

For example, given [1, 2, 1, 1, 3, 4, 0], return 1.
"""

import itertools


def maj(l):
    return sorted(
        [(k, len(list(g))) for k, g in itertools.groupby(l, key=lambda x: x)],
        key=lambda t: t[0],
        reverse=True,
    )[0][1]


print(maj([1, 2, 1, 1, 3, 4, 0]))
