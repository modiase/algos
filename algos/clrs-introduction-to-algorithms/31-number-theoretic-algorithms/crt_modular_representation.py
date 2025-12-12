#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.click
from __future__ import annotations

import random
from functools import reduce
from operator import mul

import click


def sieve_of_eratosthenes(limit: int) -> list[int]:
    """Generate primes up to limit using Sieve of Eratosthenes."""
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False

    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False

    return [i for i in range(limit + 1) if is_prime[i]]


def first_k_primes(k: int) -> tuple[int, ...]:
    """Get first k primes."""
    return tuple(sieve_of_eratosthenes(10000)[:k])


def extended_euclid(a: int, b: int) -> tuple[int, int, int]:
    """Extended Euclidean Algorithm. Returns (gcd, x, y) where gcd(a, b) = a*x + b*y."""
    if b == 0:
        return (a, 1, 0)
    gcd, x1, y1 = extended_euclid(b, a % b)
    return (gcd, y1, x1 - (a // b) * y1)


def chinese_remainder_theorem(
    residues: tuple[int, ...], moduli: tuple[int, ...]
) -> int:
    """Solve system using CRT. Assumes moduli are pairwise coprime."""
    N = reduce(mul, moduli)
    result = 0

    for a_i, n_i in zip(residues, moduli):
        N_i = N // n_i
        _, M_i, _ = extended_euclid(N_i, n_i)
        result += a_i * N_i * M_i

    return result % N


@click.group()
def cli() -> None:
    """CRT modular representation tools."""
    pass


@cli.command()
@click.option("-n", "--number", type=int, help="Number to represent")
@click.option("-s", "--seed", type=int, default=42, help="Random seed")
@click.option("-m", "--modulus", type=int, default=2**32, help="Modulus size")
def forward(number: int | None, seed: int, modulus: int) -> None:
    """Convert number to modular representation."""
    max_value = modulus - 1

    if number is None:
        random.seed(seed)
        number = random.randint(0, max_value)

    primes = first_k_primes(10)
    while reduce(mul, primes) <= max_value:
        primes = first_k_primes(len(primes) + 1)

    click.echo(f"Number: {number}")
    click.echo(f"Primes: {primes}")
    click.echo(f"Residues: {tuple(number % p for p in primes)}")


@cli.command()
@click.option("-r", "--residue", "residues", type=int, multiple=True, required=True)
@click.option("-m", "--modulus", type=int, default=2**32)
def reverse(residues: tuple[int, ...], modulus: int) -> None:
    """Convert modular representation to number."""
    k = 1
    while reduce(mul, first_k_primes(k)) <= modulus:
        k += 1

    if len(residues) > k:
        click.echo(f"Error: Too many residues. Need at most {k} for modulus {modulus}")
        return

    residues_padded = residues + (0,) * (k - len(residues))
    primes = first_k_primes(k)

    click.echo(f"Residues: {residues_padded}")
    click.echo(f"Primes: {primes}")
    click.echo(
        f"Number mod {modulus}: {chinese_remainder_theorem(residues_padded, primes) % modulus}"
    )


if __name__ == "__main__":
    cli()
