import itertools
from collections.abc import Iterable, Iterator
from typing import Final, TypeVar, cast

from tabulate import tabulate

MAX_BITS: Final[int] = 5
MAX_INT: Final[int] = 2**MAX_BITS - 1
type BitArray = tuple[int, int, int, int]


T = TypeVar("T")


def pairwise(iterable: Iterable[T]) -> zip:
    return zip(iterable, itertools.islice(iterable, 1, None))


def to_bitarray(it: Iterable[int]) -> BitArray:
    tup = tuple(it)
    if len(tup) != 5 or not all(isinstance(elem, int) for elem in tup):
        raise ValueError
    return cast(BitArray, tup)


def int_to_bitarray(n: int) -> BitArray:
    """
    Converts an integer into a big-endian bitarray.
    """
    if n > MAX_INT:
        raise ValueError(f"{MAX_INT=} exceeded")
    return to_bitarray(n >> i & 1 for i in range(MAX_BITS - 1, -1, -1))


def bin_to_gray(bitarray: BitArray) -> BitArray:
    """
    Converts a binary bitarray into a gray code bitarray.
    """
    return (
        to_bitarray([*bitarray[:msb], 1, *(i ^ j for i, j in pairwise(bitarray[msb:]))])
        if (msb := next((i for i, bit in enumerate(bitarray) if bit == 1), None))
        is not None
        else bitarray
    )


def gray_to_bin(bitarray: BitArray) -> BitArray:
    """
    Converts a gray code bitarray into a binary bitarray.
    """
    return to_bitarray(([bitarray[0], *(i ^ j for i, j in pairwise(bitarray))]))


def bitarray_to_str(bitarray: BitArray) -> str:
    """
    Converts a bitarray into a string.
    """
    return "".join(str(i) for i in bitarray)


def bitarray_to_int(bitarray: BitArray) -> int:
    """
    Converts a bitarray into an integer.
    """
    return int("".join(str(i) for i in bitarray), 2)


def gray_code_generator(n: int) -> Iterator[int]:
    current = 0
    for _ in range(1 << n):
        yield current
        parity = 0
        temp = current
        while temp:
            parity ^= temp & 1
            temp >>= 1

        if parity == 0:
            current ^= 1
        else:
            rightmost_one_pos = (current & -current).bit_length() - 1
            current ^= 1 << (rightmost_one_pos + 1)


if __name__ == "__main__":
    print(
        tabulate(
            [
                [
                    idx,
                    i,
                    bitarray_to_str(int_to_bitarray(i)),
                ]
                for idx, i in enumerate(gray_code_generator(MAX_BITS))
            ],
            headers=["idx", "gray", "gray_bits"],
            tablefmt="grid",
        )
    )
