from collections.abc import MutableSequence

from statistics.partition import partition


def _quicksort(A: MutableSequence, p: int, l: int, r: int) -> None:
    if l == r:
        return

    q = partition(A, p, l, r)
    r_left = max(l, q - 1)
    _quicksort(A, r_left, l, r_left)
    l_right = min(q + 1, r)
    _quicksort(A, r, l_right, r)


def quicksort(A):
    return _quicksort(A, (N := len(A)) - 1, 0, N - 1)
