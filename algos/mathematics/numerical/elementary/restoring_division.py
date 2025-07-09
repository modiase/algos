""" """

import pytest
from loguru import logger


def is_negative(n):
    return n < 0


def restoring_division(dividend, divisor):
    if divisor == 0:
        raise ValueError("Division by zero")

    quotient = 0
    remainder = 0
    n = dividend.bit_length()

    for i in range(n - 1, -1, -1):
        logger.trace(f"before: i: {i}, dividend: {dividend}, remainder: {remainder}")
        remainder = (remainder << 1) | ((dividend >> i) & 1)
        remainder -= divisor

        if is_negative(remainder):
            remainder += divisor
        else:
            quotient |= 1 << i
        logger.trace(f"after: i: {i}, dividend: {dividend}, remainder: {remainder}")

    return quotient, remainder


@pytest.mark.parametrize(
    "dividend, divisor, expected_quotient, expected_remainder",
    [
        (10, 3, 3, 1),
        (17, 5, 3, 2),
        (100, 7, 14, 2),
        (255, 8, 31, 7),
        (1024, 16, 64, 0),
    ],
)
def test_restoring_division(dividend, divisor, expected_quotient, expected_remainder):
    assert restoring_division(dividend, divisor) == (
        expected_quotient,
        expected_remainder,
    )


if __name__ == "__main__":
    pytest.main([__file__])
