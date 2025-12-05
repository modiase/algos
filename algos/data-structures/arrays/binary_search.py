from collections.abc import Sequence
from typing import TypeVar

S = TypeVar("S")


def binary_search(xs: Sequence[S], x: S) -> int:
    lo, hi = 0, len(xs) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if xs[mid] < x:  # pyright: ignore[reportOperatorIssue]
            lo = mid + 1
        else:
            hi = mid
    return min(lo, hi)


if __name__ == "__main__":
    test_list: Sequence[int] = [1, 2, 3, 4, 5]
    empty_list: Sequence[int] = []
    assert binary_search(test_list, 3) == 2
    assert binary_search(test_list, 1) == 0
    assert binary_search(empty_list, 5) == -1
