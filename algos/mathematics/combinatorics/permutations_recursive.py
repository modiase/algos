#!/usr/bin/env nix-shell
#! nix-shell -p "python313.withPackages(ps: [ps.loguru])"
#! nix-shell -i python3
import sys

from loguru import logger


def fact(n):
    if n < 0:
        raise ValueError(f"Invalid input: {n=}")
    return n * fact(n - 1) if n > 1 else 1


def generate_permutations(arr):
    """
    Generate all permutations using swap-based recursion.

    At each position idx, swap each element from [idx, n) into that position,
    recurse to fill remaining positions, then swap back (backtrack).

    Example for [1, 2, 3]:
        idx=0: try 1, 2, 3 in first position
        idx=1: for each, try remaining elements in second position
        idx=2: base case, emit permutation

    Time: O(n! * n) - n! permutations, O(n) to copy each
    Space: O(n) recursion depth
    """

    def generate_permutations_internal(arr, idx):
        if idx == len(arr) - 1:
            yield arr[:]

        for i in range(idx, len(arr)):
            arr[idx], arr[i] = arr[i], arr[idx]

            yield from generate_permutations_internal(arr, idx + 1)
            arr[idx], arr[i] = arr[i], arr[idx]

    yield from generate_permutations_internal(arr, 0)


def generate_permutations_unique(arr):
    """
    Generate unique permutations, skipping duplicates.

    Same as generate_permutations but skips arr[i] if that value already
    appears in arr[idx:i] - meaning we've already tried that value at
    position idx.

    The check `arr[i] in arr[idx:i]` is O(n) per iteration, making each
    level O(n²) instead of O(n). For better performance with many duplicates,
    use a set to track seen values at each level (O(1) lookup).

    Time: O(k * n) where k = number of unique permutations
          k = n! / (c1! * c2! * ...) for duplicate counts c1, c2, ...
    """

    def generate_permutations_internal(arr, idx):
        if idx == len(arr) - 1:
            yield arr[:]

        for i in range(idx, len(arr)):
            if arr[i] in arr[idx:i]:
                continue

            arr[idx], arr[i] = arr[i], arr[idx]

            yield from generate_permutations_internal(arr, idx + 1)
            arr[idx], arr[i] = arr[i], arr[idx]

    yield from generate_permutations_internal(arr, 0)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) >= 2 else 10
    total_permutations = fact(n)
    logger.info(f"Total permutations: {n=}")
    idxs = [i * total_permutations // 10 for i in range(1, 11)]
    for idx, p in enumerate(generate_permutations(list(range(n)))):
        if idx not in idxs:
            continue
        logger.info(p)
