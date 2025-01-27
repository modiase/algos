from collections.abc import Mapping, Sequence
from typing import Final, TypeAlias


VERY_LARGE_INT: Final = (2**100) ** 100

Index: TypeAlias = tuple[int, int]


def bottom_up(
    dimensions_vector: Sequence[int],
) -> tuple[Mapping[Index, int], Mapping[Index, int]]:
    L = len(dimensions_vector)
    m: Mapping[tuple[int, int], int] = {}
    s: Mapping[tuple[int, int], int] = {}

    for r in range(1, L):
        for start in range(L - r):
            idx = (start, start + r)
            if r == 1:
                m[idx], s[idx] = 0, start
            else:
                min_cost = VERY_LARGE_INT
                for i in range(1, r):
                    cost = (
                        m[(start, start + i)]
                        + m[(start + i, start + r)]
                        + (
                            dimensions_vector[start]
                            * dimensions_vector[start + i]
                            * dimensions_vector[start + r]
                        )
                    )

                    if cost < min_cost:
                        idx = (start, start + r)
                        m[idx], s[idx] = cost, i

    return m, s


if __name__ == "__main__":
    m, s = bottom_up([5, 10, 3, 12, 5, 50])
    print(f"{m=}\n{s=}")
