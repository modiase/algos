from typing import Callable, Sequence
from itertools import accumulate

from statistics.sorting.helpers import apply_permutation, identity, occurences


def countsort(A: Sequence[int], key: Callable[[int], int] = identity) -> Sequence[int]:
    """
    Performs counting sort on array A.
    """
    k = max(map(key, A))
    d = occurences(A, key)

    C = list(accumulate((d[i] if d.get(i) is not None else 0) for i in range(k + 1)))

    N = len(A)
    R = [0] * N

    for idx, elem in enumerate(reversed(A), 1):
        R[C[(k := key(elem))] - 1] = N - idx
        C[k] -= 1

    return R


if __name__ == "__main__":
    A = [10, 20, 5, 4, 4, 6]
    print(A, apply_permutation(countsort(A, key=lambda x: x % 3), A))
