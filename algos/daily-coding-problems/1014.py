"""
Given a start word, an end word, and a dictionary of valid words, find the shortest
transformation sequence from start to end such that only one letter is changed at
each step of the sequence, and each transformed word exists in the dictionary. If
there is no possible transformation, return null. Each word in the dictionary have
the same length as start and end and is lowercase.

For example, given start = "dog", end = "cat", and dictionary = {"dot", "dop", "dat",
"cat"}, return ["dog", "dot", "dat", "cat"].

Given start = "dog", end = "cat", and dictionary = {"dot", "tod", "dat", "dar"}, return
null as there is no possible transformation from dog to cat.
"""

from typing import List

d0 = [
    "dot",
    "dop",
    "dat",
    "cat",
]

d1 = ["dot", "tod", "dat", "dar"]


def distance(w1: str, w2: str) -> int:
    d = 0
    for i in range(len(w1)):
        d += int(w1[i] != w2[i])
    return d


def solve(start: str, end: str, dictionary: List[str]):
    def dfs(current: str, path: List[str]) -> List[List[str]]:
        options = [o for o in set(dictionary) - set(path) if distance(o, current) == 1]
        if not options:
            return [[]]
        elif end in options:
            return [[*path, current, end]]
        else:
            result = []
            for o in options:
                result += dfs(o, [*path, current])
            return result

    result = [r for r in dfs(start, []) if len(r) != 0]
    if len(result) == 0:
        return None
    else:
        return sorted(result, key=lambda path: len(path))[0]


assert solve("dog", "cat", d0) == ["dog", "dot", "dat", "cat"]
assert solve("dog", "cat", d1) is None
