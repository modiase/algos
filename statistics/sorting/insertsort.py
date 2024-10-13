import random as rn

from collections.abc import MutableSequence

from statistics.sorting.memory import swap


def insert(A: MutableSequence, l: int, r: int) -> None:
    idx = l
    while idx < r and A[idx] < A[r]:
        idx += 1

    swap(A, idx, r)
    idx += 1

    while idx < r:
        swap(A, idx, r)
        idx += 1


def insertsort(A: MutableSequence) -> None:
    N = len(A)
    for i in range(0, N):
        insert(A, 0, i)


def test_insertsort():
    N = 10
    I = 100
    for _ in range(I):
        rn.seed(0)
        A = [rn.randint(0, 100) for _ in range(N)]
        A_cpy = A[:]
        insertsort(A_cpy)
        sorted_A = sorted(A)
        assert A_cpy == sorted_A, f"{A_cpy}, {sorted_A}"


test_insertsort()
