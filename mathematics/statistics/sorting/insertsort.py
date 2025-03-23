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
