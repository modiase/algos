#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.pytest -p python313Packages.loguru
"""
Given two strings word1 and word2, return the minimum number of operations required to convert word1 to word2.

You have the following three operations permitted on a word:

Insert a character
Delete a character
Replace a character
"""

import sys

import pytest
from loguru import logger


class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        """
        dp algo because at each point we solve a subproblem:
        What is the edit distance word1[:i]->word2[:j]?

        base cases:

        len(i) == 0, len(j) == 0: return 0
        len(i) == 0, len(j) > 0: return len(j) (len(j) inserts)
        len(i) > 0, len(j) == 0: return len(i), (len(i) deletes)

        insert, delete, replace:

        insert -> decrement j, keep i the same
        remove -> decrement j, keep i the same
        replace -> decrement i and j
        """
        N, M = len(word1), len(word2)
        if N == 0:
            return M
        if M == 0:
            return N

        dp = [[0] * (M + 1) for _ in range(N + 1)]
        for i in range(N + 1):
            dp[i][0] = i
        for j in range(M + 1):
            dp[0][j] = j

        for i in range(1, N + 1):
            for j in range(1, M + 1):
                char1, char2 = word1[i - 1], word2[j - 1]
                logger.trace(f"char1: {char1}, char2: {char2} at {i}, {j}")
                if char1 == char2:
                    logger.trace(f"char1 == char2 at {i}, {j}")
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    logger.trace("char1 != char2 at {i}, {j}")
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

        logger.debug(f"dp: {dp}")
        return dp[N][M]


@pytest.mark.parametrize(
    "word1, word2, expected",
    [
        ("horse", "ros", 3),
        ("intention", "execution", 5),
    ],
)
def test_solution(word1: str, word2: str, expected: int):
    assert Solution().minDistance(word1, word2) == expected


if __name__ == "__main__":
    pytest.main([__file__, *sys.argv[1:]])
