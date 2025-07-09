"""
The key thing to note is the requirement that there be a >= operator.
"""

import pytest
from loguru import logger


def binary_divide(dividend, divisor):
    if divisor == 0:
        raise ValueError("Division by zero")

    quotient = 0
    remainder = 0

    for i in range(dividend.bit_length(), -1, -1):
        logger.trace(f"before: i: {i}, dividend: {dividend}, remainder: {remainder}")
        remainder = (remainder << 1) | ((dividend >> i) & 1)
        if remainder >= divisor:
            remainder -= divisor
            quotient |= 1 << i
        logger.trace(f"after: i: {i}, dividend: {dividend}, remainder: {remainder}")

    return quotient, remainder


@pytest.mark.parametrize(
    "dividend, divisor, expected",
    [
        (13, 3, (4, 1)),
    ],
)
def test_binary_divide(dividend, divisor, expected):
    assert binary_divide(dividend, divisor) == expected


if __name__ == "__main__":
    pytest.main([__file__])
