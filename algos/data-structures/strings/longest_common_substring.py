#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.pytest
from __future__ import annotations

import pytest


def build_suffix_array(s: str) -> list[int]:
    """
    Build suffix array using simple O(n^2 log n) sorting.

    Optimal construction uses Ukkonen's algorithm in O(n) time, but that's
    extremely complex. This simpler approach sorts all suffixes.
    """
    n = len(s)
    suffixes = [(s[i:], i) for i in range(n)]
    suffixes.sort()
    return [idx for _, idx in suffixes]


def lcp_array(s: str, suffix_array: list[int]) -> list[int]:
    """
    Compute longest common prefix array.

    lcp[i] = length of longest common prefix between suffix_array[i] and suffix_array[i+1].
    """
    n = len(s)
    rank = [0] * n
    for i, suffix_idx in enumerate(suffix_array):
        rank[suffix_idx] = i

    lcp = [0] * (n - 1)
    h = 0

    for i in range(n):
        if rank[i] > 0:
            j = suffix_array[rank[i] - 1]
            while i + h < n and j + h < n and s[i + h] == s[j + h]:
                h += 1
            lcp[rank[i] - 1] = h
            if h > 0:
                h -= 1

    return lcp


def longest_common_substring_suffix_array(s1: str, s2: str) -> int:
    """
    Find longest common substring using suffix array approach.

    OPTIMAL: Ukkonen's algorithm builds suffix tree in O(m+n) linear time,
    then LCS is found via DFS in O(m+n). Total: O(m+n) time and space.
    However, Ukkonen's is extremely complex to implement correctly.

    THIS IMPLEMENTATION: Uses suffix array with O((m+n)^2 log(m+n))
    construction. Simpler but slightly slower than optimal.

    Algorithm:
    1. Concatenate strings with separator: s1 + '#' + s2
    2. Build suffix array and LCP array
    3. Find max LCP between suffixes from different strings
    """
    if not s1 or not s2:
        return 0

    combined = s1 + "#" + s2
    n1 = len(s1)

    sa = build_suffix_array(combined)
    lcp = lcp_array(combined, sa)

    max_len = 0
    for i in range(len(lcp)):
        left_from_s1 = sa[i] < n1
        right_from_s1 = sa[i + 1] < n1

        if left_from_s1 != right_from_s1:
            max_len = max(max_len, lcp[i])

    return max_len


@pytest.mark.parametrize(
    "s1, s2, expected",
    [
        ("", "", 0),
        ("", "abc", 0),
        ("abc", "", 0),
        ("abc", "abc", 3),
        ("abc", "def", 0),
        ("abcde", "ace", 1),
        ("ABCDGH", "ACDGHR", 4),
        ("geeksforgeeks", "geeks", 5),
        ("abcdxyz", "xyzabcd", 4),
        ("zxabcdezy", "yzabcdezx", 6),
        ("programming", "gaming", 4),  # "ming"
        ("hello world", "world hello", 5),  # "hello" or "world"
    ],
)
def test_longest_common_substring(s1: str, s2: str, expected: int) -> None:
    assert longest_common_substring_suffix_array(s1, s2) == expected


if __name__ == "__main__":
    pytest.main([__file__])
