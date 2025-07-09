import pytest


def is_negative(n):
    return n < 0


def nonrestoring_divide(dividend, divisor):
    if divisor == 0:
        raise ValueError("Division by zero")

    quotient = 0
    remainder = 0
    n = dividend.bit_length()

    for i in range(n - 1, -1, -1):
        remainder = (remainder << 1) | ((dividend >> i) & 1)

        if remainder >= 0:
            remainder -= divisor
        else:
            remainder += divisor

        if remainder >= 0:
            quotient |= 1 << i

    if remainder < 0:
        remainder += divisor

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
def test_nonrestoring_divide(dividend, divisor, expected_quotient, expected_remainder):
    assert nonrestoring_divide(dividend, divisor) == (
        expected_quotient,
        expected_remainder,
    )


if __name__ == "__main__":
    pytest.main([__file__])
