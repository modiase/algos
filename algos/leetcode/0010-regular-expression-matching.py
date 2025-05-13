"""
10. Regular Expression Matching

Given an input string s and a pattern p, implement regular expression matching with support for '.' and '*' where:

'.' Matches any single character.
'*' Matches zero or more of the preceding element.

The matching should cover the entire input string (not partial).
"""


class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        stack = [(s, p)]
        while stack:
            s, p = stack.pop()
            if not p:
                return not s
            first_match = len(s) != 0 and p[0] in {s[0], "."}
            if len(p) >= 2 and p[1] == "*":
                stack.append((s, p[2:]))
                if first_match:
                    stack.append((s[1:], p))
            else:
                if first_match:
                    stack.append((s[1:], p[1:]))
        return False


if __name__ == "__main__":
    solution = Solution()
    assert not solution.isMatch("aa", "a")
    assert solution.isMatch("aa", "a*")
    assert solution.isMatch("ab", ".*")
    assert solution.isMatch("aab", "c*a*b")
    assert not solution.isMatch("mississippi", "mis*is*p*.")
