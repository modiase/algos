"""
Implement an algorithm to determine if a string has all unique characters.
What if you cannot use additional data structures?
"""


def is_unique(s: str) -> bool:
    return len(s) == len(set(s))


def is_unique_no_data_structures(s: str) -> bool:
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            if s[i] == s[j]:
                return False
    return True


if __name__ == "__main__":
    test_cases = [
        ("abcde", True),
        ("aabcde", False),
        ("abcde", True),
        ("", True),
        ("abcdefghijklmnopqrstuvwxyz", True),
        ("abcdefghijklmnopqrstuvwxyza", False),
    ]
    for test_case in test_cases:
        assert is_unique(test_case[0]) == test_case[1]
        assert is_unique_no_data_structures(test_case[0]) == test_case[1]
