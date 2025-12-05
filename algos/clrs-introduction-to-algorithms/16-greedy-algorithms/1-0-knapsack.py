"""
The 1-0 knapsack problem is a classic example of a problem that CANNOT be
solved by a greedy algorthm. The greedy property does not hold because taking
the most valuable object at each step may preclude choices which lead to a
globally optimal solution. This contrasts with the fractional knapsack problem
where we can always take as much of the most valuable item (in terms of value
density) as possible to greedily find the best solution.
"""

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from itertools import chain, combinations
from typing import NamedTuple


@dataclass(frozen=True, kw_only=True)
class Item:
    value: int
    weight: int


def knapsack(items: Sequence[Item], W: int) -> int:
    N = len(items)
    if N == 0 or W == 0:
        return 0
    dp = [[0] * (W + 1) for _ in range(N + 1)]

    for j in range(W + 1):
        for i in range(1, N + 1):
            if j < items[i - 1].weight:
                dp[i][j] = dp[i - 1][j]
                continue
            dp[i][j] = max(
                dp[i - 1][j], dp[i - 1][j - items[i - 1].weight] + items[i - 1].value
            )
    return dp[N][W]


def brute_force_knapsack(items: Collection[Item], W: int) -> Collection[Item]:
    return max(
        filter(
            lambda c: sum(item.weight for item in c) <= W,
            chain.from_iterable(combinations(items, i) for i in range(len(items) + 1)),
        ),
        key=lambda x: sum(item.value for item in x),
    )


class TestCase(NamedTuple):
    description: str
    items: Sequence[Item]
    W: int


test_cases = [
    TestCase(
        description="Three items, capacity 5",
        items=[
            Item(value=10, weight=2),
            Item(value=20, weight=3),
            Item(value=30, weight=4),
        ],
        W=5,
    ),
    TestCase(description="Empty items", items=[], W=10),
    TestCase(
        description="Two items, capacity 0",
        items=[Item(value=10, weight=2), Item(value=20, weight=3)],
        W=0,
    ),
    TestCase(
        description="Three items with same value, capacity 3",
        items=[
            Item(value=10, weight=1),
            Item(value=10, weight=2),
            Item(value=10, weight=3),
        ],
        W=3,
    ),
    TestCase(
        description="Three items with different values, capacity 4",
        items=[
            Item(value=5, weight=2),
            Item(value=10, weight=2),
            Item(value=15, weight=2),
        ],
        W=4,
    ),
    TestCase(
        description="Ten items with value equals weight, capacity 15",
        items=[Item(value=i, weight=i) for i in range(1, 11)],
        W=15,
    ),
    TestCase(
        description="Three items, capacity 4",
        items=[
            Item(value=10, weight=5),
            Item(value=20, weight=6),
            Item(value=30, weight=7),
        ],
        W=4,
    ),
    TestCase(
        description="Three items, capacity 5",
        items=[
            Item(value=10, weight=2),
            Item(value=20, weight=3),
            Item(value=30, weight=5),
        ],
        W=5,
    ),
    TestCase(
        description="Three items, capacity 10",
        items=[
            Item(value=100, weight=10),
            Item(value=10, weight=1),
            Item(value=20, weight=2),
        ],
        W=10,
    ),
    TestCase(
        description="Three items with two items of weight 0, capacity 5",
        items=[
            Item(value=10, weight=0),
            Item(value=20, weight=0),
            Item(value=30, weight=5),
        ],
        W=5,
    ),
]
if __name__ == "__main__":
    for idx, case in enumerate(test_cases):
        bf = brute_force_knapsack(case.items, case.W)
        expected = sum(item.value for item in bf)
        computed = knapsack(case.items, case.W)
        assert computed == expected, (
            f"{idx=}: {case.description}\n{case.items=}\n{case.W=}\n{expected=}\n{computed=}\n{bf=}\n"
        )
