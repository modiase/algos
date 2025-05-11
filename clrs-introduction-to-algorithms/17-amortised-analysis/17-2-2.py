"""
17-2-2

Repeat the analysis of the dynamic table from Exercise 17.2-1, but this time use the accounting method.
"""

from itertools import accumulate, count, dropwhile, islice
from math import log2

from tabulate import tabulate

if __name__ == "__main__":
    cost_gen = map(lambda x: x if int(log2(x)) == log2(x) else 1, count(1))
    amortized_cost_gen = map(lambda _: 3, count())

    def row_gen():
        for i, cost, amortized_cost in zip(
            count(1), accumulate(cost_gen), accumulate(amortized_cost_gen)
        ):
            yield i, cost, amortized_cost, amortized_cost - cost

    print(
        tabulate(
            (
                x
                for _, x in dropwhile(
                    lambda tup: tup[0] < 65530, enumerate(islice(row_gen(), 65536))
                )
            ),
            headers=["i", "cost", "amortized cost", "credit"],
        )
    )
