from typing import Iterable, Sequence
from itertools import accumulate, groupby


def ilen(it: Iterable[object]) -> int:
    return sum(1 for _ in it)


def counting(A: Sequence[int], k: int) -> Sequence[int]:
    """
    Performs counting sort on array A where the maximum value is k.
    """
    C = list(
        accumulate(
            (
                d[i]
                if (d := {k: ilen(it) for k, it in groupby(A)}).get(i) is not None
                else 0
            )
            for i in range(k + 1)
        )
    )

    R = [0] * len(A)

    for elem in reversed(A):
        R[C[elem] - 1] = elem
        C[elem] -= 1

    return R


if __name__ == "__main__":
    print(counting([10, 20, 5, 4, 4, 6], 20))
