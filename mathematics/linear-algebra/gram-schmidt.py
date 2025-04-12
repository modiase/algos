from collections.abc import Sequence
from itertools import combinations

import numpy as np
from more_itertools import first

Vector = np.ndarray


def gram_schmidt(vs: Sequence[Vector]) -> Sequence[Vector]:
    if len(vs) == 0:
        return ()
    d = len(first(vs))
    if not all(len(v) == d for v in vs):
        raise ValueError("All vectors must have the same dimension.")
    basis = []
    for v in vs:
        w = v - sum(np.dot(u, v) / np.dot(u, u) * u for u in basis)
        basis.append(w)
    return tuple(e / np.linalg.norm(e) for e in basis)


if __name__ == "__main__":
    vs = [np.random.randn(3) for _ in range(3)]
    basis = gram_schmidt(vs)
    print(basis)
    print([np.dot(x, y) for (x, y) in combinations(basis, 2)])
