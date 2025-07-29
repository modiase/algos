from collections import Counter
from functools import reduce
from typing import Final

ELEMENTS: Final = (
    "A",
    "B",
    "C",
    "D",
    "E",
)
N_PERMUTATIONS: Final = reduce(lambda x, y: x * y, range(len(ELEMENTS), 0, -1))


def rank(permutation: tuple[str, ...]) -> int:
    if Counter(permutation) != Counter(ELEMENTS):
        raise ValueError("Invalid permutation")

    rank = 0
    for i, element in enumerate(permutation):
        rank += ELEMENTS.index(element) * len(ELEMENTS) ** (len(permutation) - i - 1)
    return rank


def unrank(rank: int) -> tuple[str, ...]:
    if rank < 0 or rank >= len(ELEMENTS) ** len(ELEMENTS):
        raise ValueError("Invalid rank")

    permutation = []
    for i in range(len(ELEMENTS)):
        permutation.append(ELEMENTS[rank % len(ELEMENTS)])
        rank //= len(ELEMENTS)


if __name__ == "__main__":
    print(N_PERMUTATIONS)
    print(ELEMENTS)
    print(
        rank(
            (
                "A",
                "B",
                "C",
                "D",
                "E",
            )
        )
    )
    print(
        unrank(
            rank(
                (
                    "A",
                    "B",
                    "C",
                    "D",
                    "E",
                )
            )
        )
    )
    print(unrank(1))
