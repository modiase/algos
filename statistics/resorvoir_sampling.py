import math
import random
from collections.abc import Collection, Iterable
from functools import reduce
from itertools import chain
from typing import TypeVar

from tabulate import tabulate

T = TypeVar("T")


def sample(ts: Iterable[T], k: int) -> Collection[T]:
    return reduce(
        lambda acc, tup: acc + [tup[1]]
        if len(acc) < k
        else acc[:rn] + [tup[1]] + acc[rn + 1 :]
        if (rn := random.randint(0, tup[0] - 1)) < k
        else acc,
        enumerate(ts, 1),
        [],
    )


if __name__ == "__main__":
    from collections import Counter

    N = 1000000
    K = 5
    L = 20
    p = K / L
    q = 1 - p
    mu = N * p
    sigma = math.sqrt(N * p * q)

    # Proof of correctness relies on the loop invariant: on the ith iteration,
    # the probability that the ith element is in the sample is k/i.  For the i +
    # 1th iteration, the probability that a given element remains in the sample
    # is k/i * (1 - 1/(i + 1)) = k/(i + 1) and the probability that the new
    # element is added to the sample is is k/(i + 1).
    # We can compute the Z score for each element in the sample to show that it
    # since the number of runs is large enough that the CLT applies. Recall that
    # for a random variable X ~ Bin(n, p), E[X] = np and Var[X] = np(1-p).  We
    # expect Z scores close to 0.
    print(
        tabulate(
            sorted(
                (
                    (k, v, (v - mu) / sigma)
                    for k, v in Counter(
                        chain.from_iterable(sample(range(L), K) for _ in range(N))
                    ).items()
                ),
                key=lambda t: t[0],
            ),
            headers=["Number", "Count", "Z-Score"],
            tablefmt="grid",
        )
    )
