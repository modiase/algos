import random as rn
import sys
import time
from typing import Annotated
from collections.abc import Mapping, Sequence


PositiveInt = Annotated[int, lambda x: x > 0]


def generate_prices(
    seed: int, max_length: PositiveInt
) -> Mapping[PositiveInt, PositiveInt]:
    gen = rn.Random(seed)

    def _dist(i: PositiveInt) -> PositiveInt:
        return max(int((i**1.2) * 3) + gen.randint(-5, 5), 1)

    return {_n: _dist(_n) for _n in range(1, max_length + 1, 1)}


def solve(
    prices: Mapping[PositiveInt, PositiveInt],
) -> Mapping[PositiveInt, Sequence[PositiveInt]]:
    solution = {}
    _cache = {}
    N = max(prices.keys())
    for i in range(1, N + 1):
        _cache[i] = prices[i]
        solution[i] = [i]
        for j in range(1, i):
            if (new_price := _cache[i - j] + _cache[j]) > _cache[i]:
                _cache[i] = new_price
                solution[i] = solution[i - j] + solution[j]

    return solution


if __name__ == "__main__":
    seed = int(sys.argv[1] if len(sys.argv) > 1 else time.time())
    N = 10
    print(f"{seed=}")

    prices = generate_prices(seed, N)
    print(prices)

    solution = solve(prices)
    print(solution)
