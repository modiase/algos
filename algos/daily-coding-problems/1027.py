"""
Given an unsorted array of integers, find the length of the longest consecutive elements sequence.

For example, given [100, 4, 200, 1, 3, 2], the longest consecutive element sequence is [1, 2, 3, 4]. Return its length: 4.

Your algorithm should run in O(n) complexity.
Solved: ~30m
"""

from typing import List


def solution(xs: List[int]) -> int:
    s = set(xs)
    has_prev = set()
    has_next = set()

    for x in xs:
        if x - 1 in s:
            has_prev.add(x)
        if x + 1 in s:
            has_next.add(x)

    seen = set()
    longest_seq = 0
    for x in xs:
        if x in seen:
            continue
        seen.add(x)
        u = x
        uc = 0
        while u in has_next:
            uc += 1
            u += 1
            seen.add(u)
        p = x
        pc = 0
        while p in has_prev:
            pc += 1
            p -= 1
            seen.add(p)
        longest_seq = max(longest_seq, uc + pc + 1)

    return longest_seq


assert solution([100, 4, 200, 1, 3, 2]) == 4
assert solution([100, 4, 200, 5, 10, 7, 6, 9, 1, 3, 2]) == 7
assert solution([100, 4, 200, 5, 10, 7, 6, 9, 1, 3, 2, 8]) == 10
