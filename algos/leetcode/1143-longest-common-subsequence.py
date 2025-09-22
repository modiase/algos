#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.pytest -p python313Packages.loguru
"""
Given two strings text1 and text2, return the length of their longest common subsequence. If there is no common subsequence, return 0.

A subsequence of a string is a new string generated from the original string with some characters (can be none) deleted without changing the relative order of the remaining characters.

For example, "ace" is a subsequence of "abcde".
A common subsequence of two strings is a subsequence that is common to both strings.



Example 1:

Input: text1 = "abcde", text2 = "ace"
Output: 3
Explanation: The longest common subsequence is "ace" and its length is 3.
Example 2:

Input: text1 = "abc", text2 = "abc"
Output: 3
Explanation: The longest common subsequence is "abc" and its length is 3.
Example 3:

Input: text1 = "abc", text2 = "def"
Output: 0
Explanation: There is no such common subsequence, so the result is 0.


Constraints:

1 <= text1.length, text2.length <= 1000
text1 and text2 consist of only lowercase English characters.
"""

import sys

import pytest


class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        N = len(text1)
        M = len(text2)
        if N == 0 or M == 0:
            return 0

        dp = [[0] * (M + 1) for _ in range(N + 1)]
        for i in range(N - 1, -1, -1):
            for j in range(M - 1, -1, -1):
                if text2[j] == text1[i]:
                    dp[i][j] = dp[i + 1][j + 1] + 1
                else:
                    dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])

        return dp[0][0]


@pytest.mark.parametrize(
    "text1, text2, expected",
    [
        ("abcde", "ace", 3),
        ("abc", "abc", 3),
        ("abc", "def", 0),
        ("a" * 1000, "a" * 1000, 1000),
        ("a" * 1000, "b" * 1000, 0),
        ("ab" * 10, "ba" * 10, 19),
    ],
)
def test_longest_common_subsequence(text1: str, text2: str, expected: int):
    assert Solution().longestCommonSubsequence(text1, text2) == expected


if __name__ == "__main__":
    pytest.main([__file__], *sys.argv[1:])
