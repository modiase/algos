from collections.abc import Sequence
from functools import reduce
from itertools import combinations

import numpy as np
from more_itertools import first

type Vector = np.ndarray


def gram_schmidt(vs: Sequence[Vector]) -> Sequence[Vector]:
    if not all(len(v) == len(first(vs)) for v in vs):
        raise ValueError("All vectors must have the same dimension.")

    basis = reduce(
        lambda b, v: b + [v - sum(np.dot(u, v) / np.dot(u, u) * u for u in b)], vs, []
    )

    return tuple(e / np.linalg.norm(e) for e in basis)


if __name__ == "__main__":
    vs = [np.random.randn(3) for _ in range(3)]
    basis = gram_schmidt(vs)
    print(basis)
    print([np.dot(x, y) for (x, y) in combinations(basis, 2)])
