import os
import random as rn
import time
from itertools import chain, repeat, starmap
from functools import reduce
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


print(bits_to_int(reduce(xor, map(bits_of, xs))))
