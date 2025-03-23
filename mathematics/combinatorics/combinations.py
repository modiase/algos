from collections.abc import Sequence
from functools import lru_cache
from itertools import chain
from typing import Iterable, Iterator, Tuple, TypeVar

from more_itertools import all_equal, ilen

T = TypeVar("T")


def combinations(it: Iterable[T], k: int) -> Iterator[Tuple[T, ...]]:
    seq = list(it) if not isinstance(it, Sequence) else it
    def _gen() -> Iterator[Sequence[int]]:
        if k == 0:
            yield ()
        else:
            vec = list(range(k))
            yield vec
            while 1:
                vec[k - 1] += 1
                if vec[k - 1] < len(seq):
                    yield vec
                for i in range(k - 1, -1, -1):
                    if vec[i] >= len(seq) - (k - i):
                        continue
                    vec[i:] = range(vec[i] + 1, vec[i] + 1 + (k - i))
                    yield vec
                    break
                else:
                    return

    yield from (tuple(seq[i] for i in vec) for vec in _gen())


@lru_cache
def fac(n: int) -> int:
    if n < 0:
        raise ValueError("Factorial not defined for negative numbers.")
    if n == 0 or n == 1:
        return 1
    return n * fac(n - 1)


def nCr(n: int, r: int) -> int:
    return fac(n) / (fac(r) * fac(n - r))


if __name__ == "__main__":
    N = 10
    input_seq = list(range(N))
    assert all_equal(
        (ilen(combs) - c)
        for (combs, c) in (
            (combinations(input_seq, i), nCr(N, i)) for i in range(N + 1)
        )
    )
    # We can observed that the total number of combinations of all lengths
    # (including the empty combination) for N items is 2^N as a direct
    # consequence of the binomial distribution.
    print(ilen(chain.from_iterable(combinations(input_seq, i) for i in range(N + 1))))
