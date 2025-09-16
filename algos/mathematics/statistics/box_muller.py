#!/usr/bin/env nix-shell
#! nix-shell -i python3 -p python313Packages.more-itertools python313Packages.matplotlib
import random as rn
from itertools import islice
from math import cos, log, pi, sin
from typing import Callable, Iterable, Iterator

import matplotlib.pyplot as plt
from more_itertools import ilen


def random_uniform_generator() -> Iterator[float]:
    while 1:
        yield rn.random()


def random_normal_generator(
    uniform_generator_factory: Callable[[], Iterable[float]] = random_uniform_generator,
) -> Iterator[float]:
    _uniform_generator = iter(uniform_generator_factory())
    while 1:
        u1, u2 = islice(_uniform_generator, 2)
        n1 = (-2 * log(u1)) ** 0.5 * sin(2 * pi * u2)
        n2 = (-2 * log(u1)) ** 0.5 * cos(2 * pi * u2)
        yield n1
        yield n2


if __name__ == "__main__":
    N = 100000
    bin_count = 100

    uniforms = tuple(islice(random_uniform_generator(), N))
    normals = tuple(islice(random_normal_generator(lambda: uniforms), N))

    for i in range(1, 4):
        print(f"P(-{i} < Z < {i}): ", ilen(filter(lambda v: abs(v) < i, normals)) / N)

    fig, (ax_n, ax_u) = plt.subplots(1, 2)
    ax_n.hist(normals, bin_count, density=True)
    ax_u.hist(uniforms, bin_count, density=True)

    ax_u.set_title("Uniform Distribution")
    ax_u.set_xlabel("Value")
    ax_u.set_ylabel("Density")

    ax_n.set_title("Normal Distribution")
    ax_n.set_xlabel("Value")
    ax_n.set_ylabel("Density")
    plt.show()
