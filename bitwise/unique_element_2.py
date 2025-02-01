import os
import random as rn
import time
from itertools import chain, repeat, starmap
from functools import reduce
from typing import Callable, cast


rn.seed((seed := int(os.getenv("SEED", time.time()))))
print(f"{seed=}")
nums = [1, 2, 3, 4, 5]
uniq_num = rn.choice(nums)

xs = list(
    chain.from_iterable([[uniq_num], *(repeat(i, 3) for i in nums if i != uniq_num)])
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


def mod_xor(mod: int) -> Callable[[IntBits, IntBits], IntBits]:
    def _f(a: IntBits, b: IntBits):
        return cast(IntBits, tuple(starmap(lambda x, y: (x + y) % mod, zip(a, b))))

    return _f


def bits_to_int(bs: IntBits) -> int:
    return sum(starmap(lambda i, b: b * (2**i), enumerate(bs)))


print(bits_to_int(reduce(mod_xor(3), map(bits_of, xs))))
