from collections import defaultdict
from collections import deque
from itertools import chain
from typing import TypeVar
from weakref import WeakSet
from collections.abc import (
    Iterable,
    Mapping,
    MutableMapping,
)

import pytest


_T = TypeVar("_T")


def kahnsort(items: Mapping[_T, WeakSet[_T]]) -> Iterable[_T]:
    indegree: MutableMapping[_T, int] = defaultdict(lambda: 0)
    for ref in chain.from_iterable(refs for refs in items.values()):
        indegree[ref] += 1

    q = deque(items.keys() - indegree.keys())
    if not q:
        raise RuntimeError("Cycle detected: no starting node found.")

    while q:
        head = q.popleft()
        for ref in items[head]:
            indegree[ref] -= 1
            if indegree[ref] == 0:
                indegree.pop(ref)
                q.append(ref)
        yield head

    if indegree:
        raise RuntimeError(f"Cycle detected: items remaining {q=}.")


class A:
    def __init__(self, i):
        self._i = i

    def __str__(self) -> str:
        return str(self._i)

    def __repr__(self) -> str:
        return f"A({self._i})"


a = A(1)
b = A(2)
c = A(3)
d = A(4)


def test_simple_test() -> None:
    test_items = {
        a: WeakSet((b, c, d)),
        b: WeakSet((c, d)),
        c: WeakSet((d,)),
        d: WeakSet(),
    }
    assert tuple(kahnsort(test_items)) == (a, b, c, d)


def test_simple_cycle() -> None:
    with pytest.raises(RuntimeError, match=r".*Cycle detected.*"):
        tuple(
            kahnsort(
                {
                    a: WeakSet((b,)),
                    b: WeakSet((a,)),
                }
            )
        )


def test_chain_cycle() -> None:
    with pytest.raises(RuntimeError, match=r".*Cycle detected.*"):
        tuple(
            kahnsort(
                {
                    a: WeakSet((b,)),
                    b: WeakSet((c,)),
                    c: WeakSet((d,)),
                    d: WeakSet((b,)),
                }
            )
        )


def test_branching_simple() -> None:
    assert (
        result := tuple(
            kahnsort(
                {
                    a: WeakSet((b, c)),
                    b: WeakSet((d,)),
                    c: WeakSet((d,)),
                    d: WeakSet(()),
                }
            )
        )
    ) == (a, b, c, d) or result == (a, c, b, d)


def test_branching_and_cycle() -> None:
    with pytest.raises(RuntimeError, match=r".*Cycle detected.*"):
        tuple(
            kahnsort(
                {
                    a: WeakSet((b, c)),
                    b: WeakSet((d,)),
                    c: WeakSet((d,)),
                    d: WeakSet((c,)),
                }
            )
        )
