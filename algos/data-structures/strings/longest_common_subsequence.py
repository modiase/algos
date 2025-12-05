#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.pytest
from __future__ import annotations

import pytest


def longest_common_subsequence_2d(s1: str, s2: str) -> int:
    """
    Find length of longest common subsequence (non-contiguous).

    Uses 2D DP table. Time: O(m*n), Space: O(m*n).
    dp[i][j] = LCS length of s1[0:i] and s2[0:j].
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


def longest_common_subsequence_optimized(s1: str, s2: str) -> int:
    """
    Find length of longest common subsequence (non-contiguous).

    Uses space-optimized DP. Time: O(m*n), Space: O(min(m,n)).
    Only keeps current and previous row of DP table.
    """
    if len(s1) < len(s2):
        s1, s2 = s2, s1

    m, n = len(s1), len(s2)
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])

        prev, curr = curr, prev

    return prev[n]


@pytest.mark.parametrize(
    "s1, s2, expected",
    [
        ("", "", 0),
        ("", "abc", 0),
        ("abc", "", 0),
        ("abc", "abc", 3),
        ("abc", "def", 0),
        ("abcde", "ace", 3),
        ("ABCDGH", "AEDFHR", 3),
        ("AGGTAB", "GXTXAYB", 4),
        ("programming", "gaming", 6),
        ("abcdefghijk", "ecdgi", 4),
    ],
)
def test_longest_common_subsequence_2d(s1: str, s2: str, expected: int) -> None:
    assert longest_common_subsequence_2d(s1, s2) == expected


@pytest.mark.parametrize(
    "s1, s2, expected",
    [
        ("", "", 0),
        ("", "abc", 0),
        ("abc", "", 0),
        ("abc", "abc", 3),
        ("abc", "def", 0),
        ("abcde", "ace", 3),
        ("ABCDGH", "AEDFHR", 3),
        ("AGGTAB", "GXTXAYB", 4),
        ("programming", "gaming", 6),
        ("abcdefghijk", "ecdgi", 4),
    ],
)
def test_longest_common_subsequence_optimized(s1: str, s2: str, expected: int) -> None:
    assert longest_common_subsequence_optimized(s1, s2) == expected


def test_both_implementations_agree() -> None:
    """Verify both implementations produce the same results."""
    test_cases = [
        ("ABCBDAB", "BDCABA"),
        ("stone", "longest"),
        ("abcdefg", "bdfg"),
        ("xyz", "abc"),
        ("aaa", "aa"),
        ("intentionally", "executing"),
    ]

    for s1, s2 in test_cases:
        result_2d = longest_common_subsequence_2d(s1, s2)
        result_opt = longest_common_subsequence_optimized(s1, s2)
        assert result_2d == result_opt, f"Mismatch for '{s1}', '{s2}'"


if __name__ == "__main__":
    pytest.main([__file__])
