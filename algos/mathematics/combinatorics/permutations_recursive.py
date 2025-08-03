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
    def generate_permutations_internal(arr, idx):
        if idx == len(arr) - 1:
            yield arr[:]

        for i in range(idx, len(arr)):
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
