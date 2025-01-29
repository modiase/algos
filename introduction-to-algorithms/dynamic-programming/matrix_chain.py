from collections.abc import Mapping, Sequence
from typing import Final, TypeAlias


VERY_LARGE_INT: Final = (2**100) ** 100

Index: TypeAlias = tuple[int, int]


def bottom_up(
    dimensions_vector: Sequence[int],
) -> tuple[Mapping[Index, int], Mapping[Index, int | None], Mapping[Index, str]]:
    L = len(dimensions_vector)
    m: Mapping[Index, int] = {}
    s: Mapping[Index, int | None] = {}
    d: Mapping[Index, str] = {}

    for r in range(1, L):
        for start in range(L - r):
            idx = (start, start + r)
            if r == 1:
                m[idx], s[idx], d[idx] = (
                    0,
                    None,
                    f"{dimensions_vector[start]}x{dimensions_vector[start + 1]}",
                )
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
                    dim = f"{dimensions_vector[start]}x{dimensions_vector[start + r]}"

                    if cost < min_cost:
                        idx = (start, start + r)
                        m[idx], s[idx], d[idx] = (
                            cost,
                            (start + i if r != 2 else None),
                            dim,
                        )
                        min_cost = cost

    return m, s, d


if __name__ == "__main__":
    dv1 = [5, 10, 3, 12, 5, 50, 6]
    m, s, d = bottom_up(dv1)
    print(f"{dv1=}")
    for idx in m:
        print(f"{idx=}\t{m[idx]=}\t{s[idx]=}\t{d[idx]=}")
    dv2 = [5, 10, 1, 1000, 1, 50, 6]

    print(f"\n\n{dv2=}")
    m, s, d = bottom_up(dv2)
    for idx in m:
        print(f"{idx=}\t{m[idx]=}\t{s[idx]=}\t{d[idx]=}")
