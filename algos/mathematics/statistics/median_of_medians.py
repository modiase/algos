import random
from collections.abc import Callable, Iterable
from typing import TypeVar

from more_itertools import partition


def find_median(arr: list[int]) -> int:
    if len(arr) <= 5:
        # It doesn't matter that we chose the lower median here for even-length
        # arrays because we're not using this value directly, we're just using
        # it to partition the array and we can show that the lower median still
        # eliminates at least 30% of elements when partitioning which is required
        # for O(n) performance - T(n) = T(n/5) + T(7n/10) + O(n).
        return sorted(arr)[len(arr) // 2]
    else:
        medians = [find_median(arr[i : i + 5]) for i in range(0, len(arr), 5)]
        return find_median(medians)


T = TypeVar("T")
R = TypeVar("R", bound=tuple)


def chain_right(
    it1: Iterable[T],
    it2: Iterable[T],
    transform: Callable[[Iterable[T]], R],
) -> tuple[Iterable[T], R]:
    return it1, *transform(it2)


def select(arr: list[int], k: int) -> int:
    pivot = find_median(arr)
    lows, pivots, highs = map(
        tuple,
        chain_right(
            *partition(lambda x: x >= pivot, arr),
            lambda it: partition(lambda x: x > pivot, it),
        ),
    )
    if k < len(lows):
        return select(lows, k)
    elif k < len(lows) + len(pivots):
        return pivots[0]
    else:
        return select(highs, k - len(lows) - len(pivots))


if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    for _ in range(1000):
        arr = list(random.randint(0, 1000) for _ in range(2000))
        sorted_arr = sorted(arr)
        idx = random.randint(0, len(arr) - 1)
        assert (got := select(arr, idx)) == (expected := sorted_arr[idx]), (
            f"Failed: {expected=} {got=}"
        )
    print("\033[92mAll tests passed! ✅\033[0m")
