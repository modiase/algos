from collections.abc import Callable, MutableSequence
from itertools import accumulate
from typing import TypeAlias, TypeVar

from statistics.sorting.helpers import UnboundedType, apply_permutation, identity, occurences


KeyFn : TypeAlias = Callable[[UnboundedType], int]
def countsort_range(
    A: MutableSequence[UnboundedType],
    l: int,
    r: int,
    key: KeyFn = identity,
) -> None:
    """
    Performs counting sort on array A.
    """
    k = max(map(key, A))
    d = occurences(A, key)
    C = list(
        accumulate((d[i] if d.get(i) is not None else 0) for i in range(k + 1)),
    )

    N = r - l
    R : list[UnboundedType] = []

    for idx, elem in enumerate(reversed(A), 1):
        R[C[(k := key(elem))] - 1].append(A[N - idx])
        C[k] -= 1

    for idx in range(l, r):
        A[idx] = R[idx - l]

def countsort(A: MutableSequence[int], key: Callable[):
    return countsort_range()

if __name__ == "__main__":
    A = [10, 20, 5, 4, 4, 6]
    print(A, apply_permutation(countsort_range(A, key=lambda x: x % 3), A))
