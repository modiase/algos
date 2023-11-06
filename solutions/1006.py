"""
Suppose you are given two lists of n points, one list p1, p2, ..., pn on the line 
y = 0 and the other list q1, q2, ..., qn on the line y = 1. Imagine a set of n line 
segments connecting each point pi to qi. Write an algorithm to determine how many 
pairs of the line segments intersect.
"""


def solution(ps, qs):
    n = len(ps)
    count = 0
    for i in range(n):
        for j in range(n):
            if ps[j] < ps[i] and qs[i] < qs[j]:
                count += 1

    return count


assert solution((5, 1, 2), (0, 2, 4)) == 2
assert solution((1, 2, 3), (1, 2, 3)) == 0
