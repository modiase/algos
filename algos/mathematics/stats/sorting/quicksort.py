from collections.abc import MutableSequence

from statistics.partition import partition


def quicksort_range(A: MutableSequence, p: int, left: int, r: int) -> None:
    if left == r:
        return

    q = partition(A, p, left, r)
    r_left = max(left, q - 1)
    quicksort_range(A, r_left, left, r_left)
    l_right = min(q + 1, r)
    quicksort_range(A, r, l_right, r)


def quicksort(A):
    return quicksort_range(A, (N := len(A)) - 1, 0, N - 1)
