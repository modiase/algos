"""
One way to unlock an Android phone is through a pattern of swipes across a 1-9 keypad.

For a pattern to be valid, it must satisfy the following:

All of its keys must be distinct.
It must not connect two keys by jumping over a third key, unless that key has already been used.
For example, 4 - 2 - 1 - 7 is a valid pattern, whereas 2 - 1 - 7 is not.

Find the total number of valid unlock patterns of length N, where 1 <= N <= 9.
"""
import itertools


def all_sequences(n):
    return itertools.permutations(range(1, 10), n)


g = [
    [2, 4, 5],
    [1, 3, 4, 5, 6],
    [2, 5, 6],
    [1, 2, 5, 7, 8],
    [1, 2, 3, 4, 6, 7, 8, 9],
    [2, 3, 5, 8, 9],
    [4, 5, 8],
    [4, 5, 6, 7, 9],
    [5, 6, 8]
]


def path_between(start, end):
    def _next_step(visited, current):
        possible_moves = [p for p in g[current-1] if p not in visited]
        if end in possible_moves:
            return [[*visited, current, end]]
        if possible_moves == []:
            return []
        result = []
        for n in possible_moves:
            r = _next_step([*visited, current], n)
            if r:
                result += r
        return result
    return list(sorted(_next_step([], start), key=lambda x: len(x)))[0]


def sequence_is_valid(s):
    for i in range(0, len(s)-1):
        visited = s[:i]
        current = s[i]
        nxt = s[i+1]
        p = path_between(current, nxt)
        if len(p) == 2:
            continue
        cross = p[1]
        if cross not in visited:
            return False
    return True


ways = len(list((itertools.filterfalse(sequence_is_valid, all_sequences(2)))))
