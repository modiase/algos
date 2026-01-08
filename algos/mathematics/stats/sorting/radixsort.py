from typing import Sequence

from statistics.sorting.countsort import countsort
from statistics.sorting.helpers import apply_permutation


def radixsort(A: Sequence[int], k: int) -> Sequence[int]:
    print(A)
    for i in range(k):
        A = apply_permutation(countsort(A, lambda a: (a // 10**i) % 10), A)
        print(A)
    return A


if __name__ == "__main__":
    A = [947, 90, 797, 147, 954, 44, 130, 357, 696, 22]
    print(A, radixsort(A, 2))
