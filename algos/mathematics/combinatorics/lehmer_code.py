#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.pytest -p python313Packages.click -p python313Packages.tabulate
from __future__ import annotations

from collections.abc import Sequence
from functools import reduce
from typing import Final

import click
import pytest
from tabulate import tabulate

ELEMENTS: Final = ("A", "B", "C", "D", "E")
N_PERMUTATIONS: Final = reduce(lambda x, y: x * y, range(len(ELEMENTS), 0, -1))


def rank(permutation: Sequence[str], ordered_elements: Sequence[str]) -> int:
    """
    Ranking using the Lehmer code converts a permutation to an integer by
    computing a factoradic representation. For each position, count how many
    unused elements have smaller indices.

    Example: with 5 items (A,B,C,D,E), the permutation (B,C,D,A,E)
    has Lehmer code (1, 1, 1, 0, 0) because:
    - B has 1 element before it (A) -> 1
    - C has 1 element before it among remaining (A) -> 1
    - D has 1 element before it among remaining (A) -> 1
    - A has 0 elements before it among remaining -> 0
    - E has 0 elements before it among remaining -> 0

    This converts to rank: 1×4! + 1×3! + 1×2! + 0×1! + 0×0! = 24 + 6 + 2 = 32
    """
    if len(permutation) != len(ordered_elements) or set(permutation) != set(
        ordered_elements
    ):
        raise ValueError("Invalid permutation")

    available: list[str] = list(ordered_elements)
    lehmer_code = []

    for element in permutation:
        index = available.index(element)
        lehmer_code.append(index)
        available.remove(element)

    rank_value = 0
    for i, code in enumerate(lehmer_code):
        factorial = reduce(
            lambda x, y: x * y, range(len(permutation) - i - 1, 0, -1), 1
        )
        rank_value += code * factorial

    return rank_value


def unrank(rank_value: int, ordered_elements: Sequence[str]) -> tuple[str, ...]:
    """
    Unranking using the lehmer code involves converting the rank as an integer
    to a factoradic and then using the digits to select elements from a
    sequence of elements with removal.

    Example: with 5 items the rank 32 converts to
    4! * 1 + 3! * 1 + 2! * 1 + 1! * 0 + 0! * 0
    (1, 1, 1, 0, 0)
    so we select element 1 three times then element 0 (with removal)
    -> B, C, D, A, E
    """
    n_permutations = reduce(lambda x, y: x * y, range(len(ordered_elements), 0, -1))
    if rank_value < 0 or rank_value >= n_permutations:
        raise ValueError("Invalid rank")

    available: list[str] = list(ordered_elements)
    permutation = []

    for i in range(len(ordered_elements)):
        factorial = reduce(
            lambda x, y: x * y, range(len(ordered_elements) - i - 1, 0, -1), 1
        )
        index = rank_value // factorial
        permutation.append(available[index])
        available.pop(index)
        rank_value %= factorial

    return tuple(permutation)


@pytest.mark.parametrize(
    "permutation, expected_rank",
    [
        (("A", "B", "C", "D", "E"), 0),
        (("A", "B", "C", "E", "D"), 1),
        (("A", "B", "D", "C", "E"), 2),
        (("B", "A", "C", "D", "E"), 24),
        (("A", "C", "B", "D", "E"), 6),
        (("E", "D", "C", "B", "A"), 119),
    ],
)
def test_rank(permutation: tuple[str, ...], expected_rank: int) -> None:
    assert rank(permutation, ELEMENTS) == expected_rank


@pytest.mark.parametrize(
    "rank_value, expected_permutation",
    [
        (0, ("A", "B", "C", "D", "E")),
        (1, ("A", "B", "C", "E", "D")),
        (2, ("A", "B", "D", "C", "E")),
        (6, ("A", "C", "B", "D", "E")),
        (24, ("B", "A", "C", "D", "E")),
        (119, ("E", "D", "C", "B", "A")),
    ],
)
def test_unrank(rank_value: int, expected_permutation: tuple[str, ...]) -> None:
    assert unrank(rank_value, ELEMENTS) == expected_permutation


def test_rank_unrank_inverse() -> None:
    for i in range(N_PERMUTATIONS):
        assert rank(unrank(i, ELEMENTS), ELEMENTS) == i


def test_invalid_permutation() -> None:
    with pytest.raises(ValueError):
        rank(("A", "A", "C", "D", "E"), ELEMENTS)


def test_invalid_rank() -> None:
    with pytest.raises(ValueError):
        unrank(-1, ELEMENTS)
    with pytest.raises(ValueError):
        unrank(120, ELEMENTS)


cli = click.Group()


@cli.command("rank")
@click.option(
    "--permutation", "-p", required=True, help="Permutation to rank (e.g., 'CAB')"
)
@click.option(
    "--ordered-elements",
    "-e",
    required=True,
    help="Ordered sequence of elements (e.g., 'ABC')",
)
def rank_cmd(permutation: str, ordered_elements: str) -> None:
    perm_tuple = tuple(permutation)
    elements_tuple = tuple(ordered_elements)
    result = rank(perm_tuple, elements_tuple)
    click.echo(f"Elements: {ordered_elements}")
    click.echo(f"Permutation: {permutation}")
    click.echo(f"Rank: {result}")


@cli.command("unrank")
@click.option("--rank", "-r", required=True, type=int, help="Rank to unrank")
@click.option(
    "--ordered-elements",
    "-e",
    required=True,
    help="Ordered sequence of elements (e.g., 'ABC')",
)
def unrank_cmd(rank: int, ordered_elements: str) -> None:
    elements_tuple = tuple(ordered_elements)
    result = unrank(rank, elements_tuple)
    click.echo(f"Elements: {ordered_elements}")
    click.echo(f"Rank: {rank}")
    click.echo(f"Permutation: {''.join(result)}")


@cli.command()
def all_permutations() -> None:
    rows = []
    for i in range(N_PERMUTATIONS):
        perm = unrank(i, ELEMENTS)

        available: list[str] = list(ELEMENTS)
        lehmer_code = []
        for element in perm:
            index = available.index(element)
            lehmer_code.append(index)
            available.remove(element)

        rows.append([i, "".join(perm), str(lehmer_code)])

    output = f"Total permutations: {N_PERMUTATIONS}\n\n"
    output += tabulate(
        rows, headers=["Rank", "Permutation", "Factoradic"], tablefmt="simple"
    )
    click.echo_via_pager(output)


@cli.command()
def explain() -> None:
    click.echo("=== LEHMER CODE EXPLAINED ===\n")
    click.echo("Using a simple example with 3 elements: A, B, C\n")

    example_perm = ("C", "A", "B")
    elements = ("A", "B", "C")

    click.echo(f"Example permutation: {' '.join(example_perm)}\n")

    click.echo("Step 1: Build the Lehmer code (factoradic)")
    click.echo("-" * 50)

    available = list(elements)
    lehmer_code = []

    for i, element in enumerate(example_perm):
        index = available.index(element)
        lehmer_code.append(index)

        click.echo(f"Position {i}: element = {element}")
        click.echo(f"  Available elements: {available}")
        click.echo(f"  Index of {element} in available: {index}")
        click.echo(f"  Lehmer code digit: {index}")

        available.remove(element)
        click.echo(f"  Remove {element}, remaining: {available}\n")

    click.echo(f"Complete Lehmer code: {lehmer_code}\n")

    click.echo("Step 2: Convert Lehmer code to rank")
    click.echo("-" * 50)

    rank_value = 0
    for i, code in enumerate(lehmer_code):
        factorial_values = list(range(len(example_perm) - i - 1, 0, -1))
        factorial = 1
        for f in factorial_values:
            factorial *= f

        contribution = code * factorial
        rank_value += contribution

        click.echo(f"Position {i}: Lehmer digit = {code}")
        click.echo(f"  Factorial: ({len(example_perm) - i - 1})! = {factorial}")
        click.echo(f"  Contribution: {code} × {factorial} = {contribution}")
        click.echo(f"  Running total: {rank_value}\n")

    click.echo(f"Final rank: {rank_value}\n")

    click.echo("Step 3: Reverse process (unrank)")
    click.echo("-" * 50)
    click.echo(f"Starting with rank: {rank_value}\n")

    available = list(elements)
    permutation = []
    remaining_rank = rank_value

    for i in range(len(elements)):
        factorial_values = list(range(len(elements) - i - 1, 0, -1))
        factorial = 1
        for f in factorial_values:
            factorial *= f

        index = remaining_rank // factorial
        chosen = available[index]
        permutation.append(chosen)

        click.echo(f"Position {i}:")
        click.echo(f"  Available elements: {available}")
        click.echo(f"  Factorial: ({len(elements) - i - 1})! = {factorial}")
        click.echo(f"  Index: {remaining_rank} ÷ {factorial} = {index}")
        click.echo(f"  Choose element at index {index}: {chosen}")

        available.pop(index)
        remaining_rank %= factorial

        click.echo(f"  Remaining rank: {remaining_rank}")
        click.echo(f"  Remaining elements: {available}\n")

    click.echo(f"Reconstructed permutation: {' '.join(permutation)}")


@cli.command("test")
def run_tests() -> None:
    pytest.main([__file__])


if __name__ == "__main__":
    cli()
