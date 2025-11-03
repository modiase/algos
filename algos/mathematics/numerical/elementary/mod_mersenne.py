#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.loguru -p python313Packages.pytest -p python313Packages.pyvis -p python313Packages.more-itertools -p python313Packages.click
from __future__ import annotations

import random
import timeit

import click
import pytest


def mod_mersenne(value: int, prime: int) -> int:
    """Reduce an integer modulo a Mersenne prime using masks and shifts.

    Step 1: determine ``k`` such that ``prime == 2**k - 1``. Step 2: build a mask of the
    low ``k`` bits and use right-shifts of ``k`` bits to read the high chunk. Step 3:
    repeatedly fold by setting ``value = (value & mask) + (value >> k)`` until the number
    drops below ``2**k``. Because ``2**k ≡ 1 (mod prime)``, each fold discards one full
    modulus using only bit-twiddling. Step 4: map the final remainder into ``[0, prime)``
    and mirror negatives.

    Example: ``100 mod 7``. Here ``prime = 7`` implies ``k = 3``.

        100 (binary) : 1 1001 00
                        |----|^^ mask with 0b111 (low k bits)
                        high ----> (100 >> 3) == 12

        Fold 1: (100 & 0b111) + (100 >> 3) == 4 + 12 == 16
                16 (binary) :    10 000
                                |--|^^ mask with 0b111
                                high ----> (16 >> 3) == 2

        Fold 2: (16 & 0b111) + (16 >> 3) == 0 + 2 == 2  < 2**3

    The remainder ``2`` equals ``100 % 7``. Every step relies purely on masking and shifting,
    so after Numba compiles the loop the reducer becomes a tight sequence of native bit
    operations instead of Python's big-integer division and multiplication.
    """
    exponent = 0
    temp = prime
    while temp > 0:
        temp >>= 1
        exponent += 1
    mask = prime
    prime_plus_one = prime + 1
    negative = value < 0
    if negative:
        value = -value
    while value >= prime_plus_one:
        value = (value & mask) + (value >> exponent)
    if value == prime:
        value = 0
    if negative and value != 0:
        value = prime - value
    return value


@pytest.mark.parametrize(
    ("value", "prime", "expected"),
    [
        (0, 7, 0),
        (10, 7, 10 % 7),
        (127, (1 << 7) - 1, 127 % ((1 << 7) - 1)),
        ((1 << 18) + 12345, (1 << 13) - 1, ((1 << 18) + 12345) % ((1 << 13) - 1)),
        (
            (1 << 62) + (1 << 31) + 99,
            (1 << 31) - 1,
            ((1 << 62) + (1 << 31) + 99) % ((1 << 31) - 1),
        ),
    ],
)
def test_mod_mersenne(value: int, prime: int, expected: int) -> None:
    assert mod_mersenne(value, prime) == expected


def test_mod_mersenne_negative() -> None:
    prime = (1 << 5) - 1
    assert mod_mersenne(-123456, prime) == (-123456) % prime


def test_mod_mersenne_matches_builtin_random() -> None:
    prime = (1 << 31) - 1
    rng = random.Random(123)
    for _ in range(1000):
        value = rng.randrange(0, 1 << 48)
        assert mod_mersenne(value, prime) == value % prime


cli = click.Group()


@cli.command()
@click.argument("value", type=int)
@click.argument("prime", type=int)
def compute(value: int, prime: int) -> None:
    click.echo(mod_mersenne(value, prime))


@cli.command("test")
def run_tests() -> None:
    pytest.main([__file__])


@cli.command()
def benchmark() -> None:
    prime = (1 << 31) - 1
    rng = random.Random(42)
    values = [rng.randrange(0, 1 << 48) for _ in range(10_000)]
    mod_mersenne(values[0], prime)

    def run_mod_mersenne() -> None:
        for value in values:
            mod_mersenne(value, prime)

    def run_builtin() -> None:
        for value in values:
            value % prime  # pyright: ignore[reportUnusedExpression]

    for _ in range(5):
        run_mod_mersenne()  # pyright: ignore[reportUnusedCallResult]
        run_builtin()  # pyright: ignore[reportUnusedCallResult]

    mod_time = timeit.timeit(run_mod_mersenne, number=1_000)
    builtin_time = timeit.timeit(run_builtin, number=1_000)
    click.echo(f"mod_mersenne: {mod_time:.4f}s")
    click.echo(f"builtin %:   {builtin_time:.4f}s")


if __name__ == "__main__":
    cli()
