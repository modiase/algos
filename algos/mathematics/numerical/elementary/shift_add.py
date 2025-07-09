"""
Simplest multiplication algorithm.
"""

import pytest


def shift_add(a: int, b: int) -> int:
    return sum(a << i for i in range(b.bit_length()) if b & (1 << i))


@pytest.mark.parametrize(
    "a, b, expected",
    [
        (1, 1, 1),
        (1, 2, 2),
        (2, 1, 2),
        (2, 2, 4),
        (2, 3, 6),
        (3, 2, 6),
        (3, 3, 9),
        (3, 4, 12),
        (4, 3, 12),
    ],
)
def test_shift_add(a: int, b: int, expected: int):
    assert shift_add(a, b) == expected


if __name__ == "__main__":
    pytest.main([__file__])
