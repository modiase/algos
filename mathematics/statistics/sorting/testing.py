import random as rn
from collections.abc import Callable, MutableSequence


def test_sort(sort_fn: Callable[[MutableSequence], None]):
    N = 10
    I = 100
    for _ in range(I):
        A = [rn.randint(0, 100) for _ in range(N)]
        A_cpy = A[:]
        sort_fn(A_cpy)
        sorted_A = sorted(A)
        assert A_cpy == sorted_A, f"{A_cpy}, {sorted_A}"
