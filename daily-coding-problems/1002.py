"""
Good morning! Here's your coding interview problem for today.

This problem was asked by Amazon.

Given a sorted array, find the smallest positive integer that is not the sum of a subset of the array.

For example, for the input [1, 2, 3, 10], you should return 7.

Do this in O(N) time.
"""
import itertools


def flatten(l):
    return [item for sublist in l for item in sublist]


def min_sum(l):
    g = itertools.chain(((sum(t) for t in itertools.combinations(l, n)))
                        for n in range(1, len(l)))
    ss = [subitem for item in (i for i in g) for subitem in item]
    return min([s + 1 for s in ss if (s+1) not in ss])


print(min_sum([1, 2, 3, 10]))
