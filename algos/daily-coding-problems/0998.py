"""
Problem:

Given an undirected graph represented as an adjacency matrix and an integer k,
write a function to determine whether each vertex in the graph can be colored
such that no two adjacent vertices share the same color using at most k colors.

"""

import itertools
from collections.abc import Sequence


def solve(m: Sequence[Sequence[int]], k: Sequence[str]):
    c: list[str] = [k[0] for _ in range(0, len(m))]
    h: list[list[str]] = []

    def _solve(m: Sequence[Sequence[int]], c: list[str]):
        nonlocal h
        for self_idx, node in enumerate(m, 0):
            for idx, _ in enumerate(node, 0):
                if c[self_idx] == c[idx] and (idx != self_idx):
                    color_idx = k.index(c[self_idx])
                    next_color_idx = color_idx + 1 if (color_idx + 1) < len(k) else 0
                    c1 = c[0:self_idx] + [k[next_color_idx]] + c[self_idx + 1 : len(c)]
                    c2 = c[0:idx] + [k[next_color_idx]] + c[idx + 1 : len(c)]
                    if c1 in h and c2 in h:
                        return None
                    branch1, branch2 = None, None
                    if c1 not in h:
                        h = [*h, c1]
                        branch1 = _solve(m, c1)
                    if c2 not in h:
                        h = [*h, c2]
                        branch2 = _solve(m, c2)
                    return branch1 or branch2
        return c

    return _solve(m, c)


def can_color(m, k):
    return solve(m, k) is not None


m1 = [[1, 1, 0, 0], [1, 1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 1]]
cs = ([f"c{n}" for n in range(1, length)] for length in itertools.count(start=2))

result = None
while not can_color(m1, c := next(cs)):
    pass
print(c)
