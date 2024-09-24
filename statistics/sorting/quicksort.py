from collections.abc import MutableSequence

from statistics.partition import partition


def quicksort_range(A: MutableSequence, p: int, l: int, r: int) -> None:
    if l == r:
        return

    q = partition(A, p, l, r)
    r_left = max(l, q - 1)
    quicksort_range(A, r_left, l, r_left)
    l_right = min(q + 1, r)
    quicksort_range(A, r, l_right, r)


def quicksort(A):
    return quicksort_range(A, (N := len(A)) - 1, 0, N - 1)
