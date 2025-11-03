import math
from collections.abc import Iterator
from itertools import chain
from typing import Tuple, TypeVar

from more_itertools import ilen

T = TypeVar("T")


def all_permutations(n: int, k: int) -> Iterator[Tuple[int, ...]]:
    if k == 0:
        yield ()
        return

    if k > n:
        return

    stack: list[tuple[tuple[int, ...], list[int]]] = [((), list(range(n)))]

    while stack:
        prefix, remaining = stack.pop()

        if len(prefix) == k:
            yield prefix
            continue

        for i in range(len(remaining) - 1, -1, -1):
            new_prefix = prefix + (remaining[i],)
            new_remaining = remaining[:i] + remaining[i + 1 :]
            stack.append((new_prefix, new_remaining))


if __name__ == "__main__":
    N = 3
    tot_permutations = ilen(
        chain.from_iterable(all_permutations(N, i) for i in range(N + 1))
    )
    # The total number of permutations of all lengths of N items is
    #  floor(N! * e).
    assert tot_permutations == math.floor(math.e * math.factorial(N))
