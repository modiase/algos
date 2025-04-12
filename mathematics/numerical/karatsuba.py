def karatsuba(x, y):
    """
    Karatsuba algorithm for integer multiplication.

    The naive approach to multiply two numbers is to multiply each digit of the
    first number by each digit of the second number and then sum the results.
    This is a O(n^2) algorithm due to the recurence relation:

    T(n) = 4T(n/2) + O(n)
    where n is the number of digits in the number. This arises from splitting
    the digits into two halves and then performing the four multiplication
    subproblems.

    The Karatsuba algorithm is a O(n^log_2(3)) algorithm.

    It achieves this by reducing the number of multiplications from four to
    three to achieve the following recurrence relation:

    T(n) = 3T(n/2) + O(n)
    """
    if x < 10 or y < 10:
        return x * y

    n = max(len(str(x)), len(str(y)))
    m = n // 2

    x_high, x_low = divmod(x, 10**m)
    y_high, y_low = divmod(y, 10**m)

    z0 = karatsuba(x_low, y_low)
    z1 = karatsuba(x_low + x_high, y_low + y_high)
    z2 = karatsuba(x_high, y_high)

    return z2 * 10 ** (2 * m) + (z1 - z2 - z0) * 10**m + z0


if __name__ == "__main__":
    numbers = [
        (1234, 5678),
        (12345678, 98765432),
        (1234567890, 9876543210),
    ]
    for x, y in numbers:
        assert karatsuba(x, y) == x * y
