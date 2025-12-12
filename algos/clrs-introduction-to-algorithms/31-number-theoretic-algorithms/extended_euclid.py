#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.pytest
from __future__ import annotations

import pytest


def extended_euclid(a: int, b: int) -> tuple[int, int, int]:
    """
    Extended Euclidean Algorithm.

    Returns (gcd, x, y) where gcd(a, b) = a*x + b*y.

    Time: O(log min(a, b)), Space: O(log min(a, b))
    """
    if b == 0:
        return (a, 1, 0)

    gcd, x1, y1 = extended_euclid(b, a % b)
    return (gcd, y1, x1 - (a // b) * y1)


def modular_multiplicative_inverse(a: int, n: int) -> int | None:
    """
    Compute modular multiplicative inverse of a modulo n.

    Returns x where (a * x) ≡ 1 (mod n), or None if gcd(a, n) ≠ 1.

    Time: O(log min(a, n))
    """
    gcd, x, _ = extended_euclid(a, n)

    if gcd != 1:
        return None

    return x % n


@pytest.mark.parametrize(
    "a, b, expected_gcd",
    [
        (30, 21, 3),
        (99, 78, 3),
        (252, 105, 21),
        (17, 13, 1),
        (1071, 462, 21),
        (0, 5, 5),
        (5, 0, 5),
        (1, 1, 1),
        (48, 18, 6),
        (100, 35, 5),
    ],
)
def test_extended_euclid_gcd(a: int, b: int, expected_gcd: int) -> None:
    gcd, _, _ = extended_euclid(a, b)
    assert gcd == expected_gcd


@pytest.mark.parametrize(
    "a, b",
    [
        (30, 21),
        (99, 78),
        (252, 105),
        (17, 13),
        (1071, 462),
        (48, 18),
        (100, 35),
    ],
)
def test_extended_euclid_bezout_identity(a: int, b: int) -> None:
    gcd, x, y = extended_euclid(a, b)
    assert a * x + b * y == gcd


@pytest.mark.parametrize(
    "a, n, has_inverse",
    [
        (3, 7, True),
        (5, 11, True),
        (2, 4, False),
        (6, 9, False),
        (7, 26, True),
        (15, 26, True),
        (4, 8, False),
    ],
)
def test_modular_multiplicative_inverse(a: int, n: int, has_inverse: bool) -> None:
    inverse = modular_multiplicative_inverse(a, n)

    if has_inverse:
        assert inverse is not None
        assert (a * inverse) % n == 1
    else:
        assert inverse is None


if __name__ == "__main__":
    pytest.main([__file__])
