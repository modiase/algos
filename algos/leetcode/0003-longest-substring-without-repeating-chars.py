#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.pytest
from __future__ import annotations

import pytest


class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        N = len(s)
        if N == 0:
            return 0
        left, right = 0, 1
        letters = {s[0]: 0}
        longest = 1
        while right < N:
            if s[right] in letters and left <= letters[s[right]]:
                left = letters[s[right]] + 1
            else:
                longest = max(right + 1 - left, longest)
            letters[s[right]] = right
            right += 1

        return longest


@pytest.mark.parametrize(
    "s, expected",
    [
        ("", 0),
        ("a", 1),
        ("ab", 2),
        ("abcabcbb", 3),
        ("bbbbb", 1),
        ("pwwkew", 3),
        ("abcdef", 6),
        ("aab", 2),
        ("dvdf", 3),
        ("anviaj", 5),
        ("abba", 2),
        (" ", 1),
        ("au", 2),
        ("tmmzuxt", 5),
    ],
)
def test_length_of_longest_substring(s: str, expected: int) -> None:
    solution = Solution()
    assert solution.lengthOfLongestSubstring(s) == expected


if __name__ == "__main__":
    pytest.main([__file__])
