#!/usr/bin/env nix-shell
#! nix-shell -i python3 -p python313
import random


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


def partition(arr: list[int], pivot: int) -> tuple[list[int], list[int], list[int]]:
    lows = []
    pivots = []
    highs = []
    for x in arr:
        if x < pivot:
            lows.append(x)
        elif x == pivot:
            pivots.append(x)
        else:
            highs.append(x)
    return lows, pivots, highs


def select(arr: list[int], k: int) -> int:
    pivot = find_median(arr)
    lows, pivots, highs = partition(arr, pivot)

    if k < len(lows):
        return select(lows, k)
    elif k < len(lows) + len(pivots):
        return pivot
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
