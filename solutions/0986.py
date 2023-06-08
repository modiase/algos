"""
Given an integer n, return the length of the longest consecutive run of 1s in its binary representation.

For example, given 156, you should return 3.
"""
import math


def longest_run(l, c):
    m = 0
    counter = 0
    for c0 in l:
        if c0 == c:
            counter += 1
        else:
            m = max(counter, m)
            counter = 0
    m = max(counter, m)
    return m


def to_bin(n):
    s = ''
    i = 0
    while (p := math.pow(2, i)) < n:
        i += 1
        if int(p) & n:
            s += '1'
        else:
            s += '0'
    return ''.join(reversed(s))


assert to_bin(156), '1' == 3
