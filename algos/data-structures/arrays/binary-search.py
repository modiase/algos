from collections.abc import Sequence
from typing import Protocol, Self, TypeVar


class SupportsLessThan(Protocol):
    def __lt__(self, other: Self) -> bool: ...


S = TypeVar("S", bound=SupportsLessThan)


def binary_search(xs: Sequence[S], x: S) -> int:
    lo, hi = 0, len(xs) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if xs[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return min(lo, hi)


if __name__ == "__main__":
    assert binary_search([1, 2, 3, 4, 5], 3) == 2
    assert binary_search([1, 2, 3, 4, 5], 1) == 0
    assert binary_search([], 5) == -1
