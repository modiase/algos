"""
Given an array of integers where every element appears twice except for one element, find the unique element.

Example:
Input: [1, 2, 3, 4, 5, 1, 2, 3, 4]
Output: 5

"""

import os
import random as rn
import time
from functools import reduce
from itertools import chain, repeat, starmap
from typing import cast

rn.seed((seed := int(os.getenv("SEED", time.time()))))
print(f"{seed=}")
nums = [1, 2, 3, 4, 5]
uniq_num = rn.choice(nums)

xs = list(
    chain.from_iterable([[uniq_num], *(repeat(i, 2) for i in nums if i != uniq_num)])
)
rn.shuffle(xs)
print(xs)


IntBits = tuple[
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
]


def bits_of(i: int) -> IntBits:
    return cast(IntBits, tuple(i >> n & 1 for n in range(32)))


def xor(a: IntBits, b: IntBits) -> IntBits:
    return cast(IntBits, tuple(starmap(lambda x, y: int(x != y), zip(a, b))))


def bits_to_int(bs: IntBits) -> int:
    return sum(starmap(lambda i, b: b * (2**i), enumerate(bs)))


# The solution relies on the fact that A XOR A = 0 two identical numbers results
# in 0.  Therefore, when we xor all the numbers in the array, the numbers that
# appear twice will cancel each other out, and we will be left with the unique
# number.
print(bits_to_int(reduce(xor, map(bits_of, xs))))
# It may be generally tempting to try and generalise the use of XOR to find
# unmached elements but it is important to note that it is possible for a
# combination of numbers to coincidentally have an XOR that is 0.