"""
Given two strings, write a method to decide if one is a permutation of the other.
"""

def is_permutation(s1: str, s2: str) -> bool:
    return sorted(s1) == sorted(s2)

if __name__ == '__main__':
    test_cases = [
        ('abcde', 'edcba', True),
        ('abcde', 'edcbaf', False),
        ('', '', True),
        ('a', '', False),
        ('', 'b', False),
        ('a', 'a', True),
        ('a', 'b', False),
        ('abcde', 'edcbaf', False),
    ]
    for test_case in test_cases:
        assert is_permutation(test_case[0], test_case[1]) == test_case[2]
