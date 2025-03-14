from collections import Counter
from operator import itemgetter
from typing import Callable, Sequence, TypeVar
from itertools import accumulate


UnboundedType = TypeVar("UnboundedType")


def apply_permutation(
    P: Sequence[int], S: Sequence[UnboundedType]
) -> Sequence[UnboundedType]:
    return list(itemgetter(*P)(S))


def identity(x: int) -> int:
    return x


def countsort(A: Sequence[int], key: Callable[[int], int] = identity) -> Sequence[int]:
    """
    Performs counting sort on array A.
    """
    k = max(map(key, A))
    d = Counter(map(key, A))

    C = list(accumulate(d.get(i, 0) for i in range(k + 1)))

    N = len(A)
    R = [0] * N

    for idx, elem in enumerate(reversed(A), 1):
        R[C[(k := key(elem))] - 1] = N - idx
        C[k] -= 1

    return R


if __name__ == "__main__":
    A = [10, 20, 5, 4, 4, 6]
    assert (countsorted := apply_permutation(countsort(A), A)) == sorted(countsorted)
    assert (
        countsorted := apply_permutation(countsort(A, key=lambda x: x % 3), A)
    ) == sorted(countsorted, key=lambda x: x % 3)
