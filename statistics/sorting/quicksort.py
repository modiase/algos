from collections.abc import MutableSequence
import random as rn

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


def test_quicksort():
    N = 10
    I = 100
    for _ in range(I):
        A = [rn.randint(0, 100) for _ in range(N)]
        A_cpy = A[:]
        quicksort(A_cpy)
        sorted_A = sorted(A)
        assert A_cpy == sorted_A, f"{A_cpy}, {sorted_A}"
