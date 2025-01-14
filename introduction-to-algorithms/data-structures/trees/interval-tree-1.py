from collections.abc import Collection, Iterable, Iterator, MutableSequence
import random as rn
from dataclasses import dataclass
from itertools import chain, dropwhile, islice, takewhile
from time import time
from typing import NamedTuple, Self

N, L, U = 7, 0, 1000

seed = int(time())
print(f"{seed=}")
rn.seed(seed)


def ilen(i: Iterable[object]) -> int:
    return sum(1 for _ in i)


class Interval(NamedTuple):
    start: float
    end: float

    @property
    def mid(self) -> float:
        return (self.end + self.start) / 2

    def contains(self, value: float) -> bool:
        return value <= self.end and value >= self.start

    def overlaps(self, other: Self) -> bool:
        return self.contains(other.start) or self.contains(other.end)


def clip(value: float, minimum: float, maximum: float) -> float:
    return max(min(value, maximum), minimum)


intervals = tuple(
    (
        Interval(
            *sorted(((l := rn.randint(L, U)), clip(l + rn.normalvariate(0, 1), L, U)))
        )
        for _ in range(N)
    )
)


def _median(i: Iterator[float]) -> float | None:
    head = next(i, None)
    if head is None:
        return None
    s = sorted(chain((head,), i))
    N = len(s)
    return s[N // 2] if N % 2 == 0 else (s[N // 2] + s[(N // 2) - 1]) / 2


def median_of_medians(i: Iterator[float]) -> float | None:
    medians = []
    while True:
        median = _median(islice(i, 5))
        if median is None:
            break
        medians.append(median)
    return _median(iter(medians))


@dataclass
class Node:
    @dataclass
    class Intervals:
        _intervals: list[Interval]

        @classmethod
        def of(cls, intervals: Collection[Interval]):
            return cls(_intervals=list(intervals))

        @property
        def by_start(self):
            return sorted(self._intervals, key=lambda i: i[0])

        @property
        def by_end(self):
            return sorted(self._intervals, key=lambda i: i[1])

        def __len__(self):
            return len(self._intervals)

    mid: float
    left: Self | None
    right: Self | None
    intervals: Intervals

    def yield_bfs(self) -> Iterator[Self]:
        stack: MutableSequence[Self] = [self]

        while stack:
            current = stack.pop()
            yield current
            if current.left is not None:
                stack.append(current.left)
            if current.right is not None:
                stack.append(current.right)


    def yield_dfs(self) -> Iterator[Self]:
        stack: MutableSequence[Self] = []
        current = self
        last_visited = None

        while stack or current:
            if current:
                stack.append(current)
                current = current.left
                continue

            top = stack[-1]

            if top.right and top.right != last_visited:
                current = top.right
            else:
                yield top
                last_visited = stack.pop()
                current = None

    @staticmethod
    def compute_mid(intervals: Collection[Interval]) -> float:
        if (result := median_of_medians(i.mid for i in intervals)) is not None:
            return result
        raise RuntimeError("Empty intervals passed")

    @staticmethod
    def partition(
        m: float, intervals: Collection[Interval]
    ) -> tuple[Collection[Interval], Collection[Interval], Collection[Interval]]:
        l, c, r = [], [], []
        for interval in intervals:
            if interval.end < m:
                l.append(interval)
            elif interval.start > m:
                r.append(interval)
            else:
                c.append(interval)
        return l, c, r

    @classmethod
    def of(cls, intervals: Collection[Interval]) -> Self | None:
        if len(intervals) == 0:
            return None

        mid = cls.compute_mid(intervals)
        l, c, r = cls.partition(mid, intervals)
        return cls(
            mid=mid, left=cls.of(l), right=cls.of(r), intervals=cls.Intervals.of(c)
        )

    def __str__(self):
        return f"Node({len(self.intervals)}, left={self.left}, right={self.right})"

    def intersect(self, interval: Interval) -> Collection[Interval]:
        result: MutableSequence[Interval] = []
        stack: MutableSequence[Node] = [self]
        while stack:
            current = stack.pop()
            if current.mid <= interval.end and (right := current.right) is not None:
                stack.append(right)
            if interval.start <= current.mid and (left := current.left) is not None:
                stack.append(left)

            result.extend(
                {
                    i
                    for i in chain(
                        takewhile(
                            lambda other: other.start <= interval.end,
                            current.intervals.by_start,
                        ),
                        dropwhile(
                            lambda other: other.end < interval.start,
                            current.intervals.by_end,
                        ),
                    )
                    if interval.overlaps(i)
                }
            )

        return result


node = Node.of(intervals)
assert node is not None
interval = Interval(1, 100)
intersecting = node.intersect(interval)
print(list(n.intervals for n in node.yield_dfs()))
print(list(n.intervals for n in node.yield_bfs()))
