from operator import itemgetter
from typing import Callable, Hashable, Iterable, Mapping, Sequence, TypeVar

UnboundedType = TypeVar("UnboundedType")


def ilen(it: Iterable[object]) -> int:
    return sum(1 for _ in it)


def identity(i: int) -> int:
    return i


def apply_permutation(
    P: Sequence[int], S: Sequence[UnboundedType]
) -> Sequence[UnboundedType]:
    return list(itemgetter(*P)(S))


def occurences(
    it: Iterable[UnboundedType], key: Callable[[UnboundedType], Hashable] = hash
) -> Mapping[Hashable, int]:
    d = {}
    for elem in it:
        k = key(elem)
        d[k] = d.setdefault(k, 0) + 1
    return d
