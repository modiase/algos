from typing import Final

from more_itertools import pairwise
from tabulate import tabulate

MAX_BITS: Final[int] = 5
MAX_INT: Final[int] = 2**MAX_BITS - 1
BitArray = tuple[int, int, int, int]


def int_to_bitarray(n: int) -> BitArray:
    """
    Converts an integer into a big-endian bitarray.
    """
    if n > MAX_INT:
        raise ValueError(f"{MAX_INT=} exceeded")
    return tuple(n >> i & 1 for i in range(MAX_BITS - 1, -1, -1))


def bin_to_gray(bitarray: BitArray) -> BitArray:
    """
    Converts a binary bitarray into a gray code bitarray.
    """
    return (
        tuple([*bitarray[:msb], 1, *(i ^ j for i, j in pairwise(bitarray[msb:]))])
        if (msb := next((i for i, bit in enumerate(bitarray) if bit == 1), None))
        is not None
        else bitarray
    )


def gray_to_bin(bitarray: BitArray) -> BitArray:
    """
    Converts a gray code bitarray into a binary bitarray.
    """
    return tuple(i ^ j for i, j in pairwise(bitarray))


def bitarray_to_str(bitarray: BitArray) -> str:
    """
    Converts a bitarray into a string.
    """
    return "".join(str(i) for i in bitarray)


if __name__ == "__main__":
    print(
        tabulate(
            [
                [
                    i,
                    bitarray_to_str(bits := int_to_bitarray(i)),
                    bitarray_to_str(bin_to_gray(bits)),
                ]
                for i in range(MAX_INT + 1)
            ],
            headers=["int", "bin", "gray"],
            tablefmt="grid",
        )
    )
