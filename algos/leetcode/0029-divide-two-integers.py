"""
Given two integers dividend and divisor, divide two integers without using
multiplication, division, and mod operator.

The integer division should truncate toward zero, which means losing its
fractional part. For example, 8.345 would be truncated to 8, and -2.7335 would
be truncated to -2.

Return the quotient after dividing dividend by divisor.

Note: Assume we are dealing with an environment that could only store integers
within the 32-bit signed integer range: [−231, 231 − 1]. For this problem, if
the quotient is strictly greater than 231 - 1, then return 231 - 1, and if the
quotient is strictly less than -231, then return -231.
"""

import ctypes

import pytest


class Solution:
    def get_msb(self, n: int) -> int:
        shift = 1
        n = n if n > 0 else -n
        while n >> shift:
            n |= n >> shift
            shift <<= 1

        return n - (n >> 1)

    def subtract(self, a: int, b: int) -> int:
        return ctypes.c_int32(a + ctypes.c_int32(~b + 1).value).value

    def is_pow2(self, n) -> int | None:
        for i in range(32):
            if n == (1 << i):
                return i
        return None

    def divide(self, dividend: int, divisor: int) -> int:
        MASK = (1 << 32) - 1
        MAX_INT = (1 << 31) - 1

        def sign_and_clip(sign, n):
            if sign == 1:
                return min(n, MAX_INT)
            else:
                return -min(n, MAX_INT + 1)

        sign = 1 - 2 * int((dividend < 0) != (divisor < 0))
        dividend = dividend if dividend > 0 else -dividend
        divisor = divisor if divisor > 0 else -divisor
        if not dividend ^ divisor:
            return sign

        if (p := self.is_pow2(divisor)) is not None:
            return sign_and_clip(sign, ((dividend >> p) & MASK))

        msb_dividend = self.get_msb(dividend)
        msb_divisor = self.get_msb(divisor)

        acc = 0
        while msb_dividend >= msb_divisor:
            shift = 0
            while msb_dividend > (msb_divisor << (shift + 1)):
                shift += 1

            dividend = self.subtract(dividend, divisor << shift)
            msb_dividend = self.get_msb(dividend)
            if dividend >= 0:
                acc += 1 << shift
        return sign_and_clip(sign, (acc & MASK))


@pytest.mark.parametrize(
    "dividend, divisor, expected",
    [
        (10, 3, 3),  # Basic positive division
        (7, -3, -2),  # One negative number
        (-7, -3, 2),  # Both negative numbers
        (-2147483648, -1, 2147483647),  # Max int edge case
        (-2147483648, 1, -2147483648),  # Min int edge case
        (2147483647, 1, 2147483647),  # Max int divided by 1
        (1024, 2, 512),  # Power of 2 divisor
        (0, 5, 0),  # Zero dividend
        (10, 1, 10),  # Division by 1
        (-2147483648, -2147483648, 1),  # Same numbers
        (2147483647, 2147483647, 1),  # Same large numbers
    ],
)
def test_divide(dividend, divisor, expected):
    assert Solution().divide(dividend, divisor) == expected


if __name__ == "__main__":
    pytest.main([__file__])
