import random as rn
import sys
import time
from collections.abc import Mapping, Sequence


def generate_prices(seed: int, max_length: int) -> Mapping[int, int]:
    gen = rn.Random(seed)

    def _dist(i: int) -> int:
        return max(int((i**1.2) * 3) + gen.randint(-5, 5), 1)

    return {_n: _dist(_n) for _n in range(1, max_length + 1, 1)}


def solve(
    prices: Mapping[int, int],
    cost_of_cut: int,
) -> Mapping[int, Sequence[tuple[int, int]]]:
    solution = {}
    _cache = {}
    N = max(prices.keys())
    for i in range(1, N + 1):
        _cache[i] = prices[i]
        solution[i] = ([i], prices[i])
        for j in range(1, i):
            if (new_price := _cache[i - j] + _cache[j] - cost_of_cut) > _cache[i]:
                _cache[i] = new_price
                solution[i] = (solution[i - j][0] + solution[j][0], _cache[i])

    return solution


def print_solution(solution: Mapping[int, Sequence[tuple[int, int]]]) -> None:
    print(f"{'Length':<10} {'Splits':<30} {'Value':<10}")
    print("-" * 52)
    for length, (cuts, value) in solution.items():
        cuts_str = ", ".join(map(str, cuts))
        print(f"{length:<10} {cuts_str:<30} {value:<10}")
    print()


if __name__ == "__main__":
    seed = int(sys.argv[1] if len(sys.argv) > 1 else time.time())
    N = 10
    print(f"{seed=}")

    prices = generate_prices(seed, N)
    print(f"{prices=}")
    print()

    for cost_of_cut in range(0, 11, 2):
        solution = solve(prices, cost_of_cut)
        print(f"{cost_of_cut=}")
        print_solution(solution)
