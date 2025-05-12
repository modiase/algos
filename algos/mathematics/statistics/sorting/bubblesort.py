from collections.abc import MutableSequence

from statistics.sorting.memory import swap


def bubblesort(A: MutableSequence) -> None:
    N = len(A)
    for start in range(1, N):
        for i in range(start, 0, -1):
            if A[i] < A[i - 1]:
                swap(A, i - 1, i)
                continue
            break
