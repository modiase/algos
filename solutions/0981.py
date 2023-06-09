"""
Describe an algorithm to compute the longest increasing subsequence of an array of numbers in O(n log n) time.

Finished: 24m
"""


import itertools


def is_subset(s, a):
    for offset in range(len(a)-len(s)):
        if s == a[offset:offset+len(s)]:
            return True
    return False


def find_longest_increasing_subsequence(a):
    sa = list(sorted(a))
    for l in range(len(a)-2, 0, -1):
        for offset in range(len(a)-l):
            subset = sa[offset:offset+l+1]
            if is_subset(subset, a):
                return len(subset)
    return 1


assert (find_longest_increasing_subsequence(
    [1, 2, 3, 4, 5, 1, 0, 1, 0, 1, 0])) == 5
