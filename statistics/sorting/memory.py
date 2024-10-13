from collections.abc import MutableSequence


def swap(A: MutableSequence, i: int, j: int) -> None:
    if i == j:
        return
    tmp = A[i]
    A[i] = A[j]
    A[j] = tmp
