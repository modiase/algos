#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.pytest -p python313Packages.click
"""
Continuation Monad

A mini-language for defining execution flows. Using << for bind and __call__
to execute, we separate logic definition from execution.

    Cont r a  ~  (a -> r) -> r

Short-circuiting happens by not calling k. Since k contains the entire future
of the program, not calling it makes the rest vanish - no if/else needed.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Generic, TypeVar

import click

R = TypeVar("R")
A = TypeVar("A")
B = TypeVar("B")


class Cont(Generic[R, A]):
    """
    Continuation monad wrapping a CPS computation.

    Cont[R, A] represents a computation that, given a continuation (A -> R),
    produces an R. The continuation may be called with a result, or the
    computation may return early to short-circuit the chain.
    """

    def __init__(self, run: Callable[[Callable[[A], R]], R]) -> None:
        self.run = run

    @staticmethod
    def pure(x: A) -> Cont[R, A]:
        """Lift a value into the continuation monad."""
        return Cont(lambda k: k(x))

    def __lshift__(self, f: Callable[[A], Cont[R, B]]) -> Cont[R, B]:
        """
        Bind (<<): hooks the next step into the current continuation.

        Returns a new Cont that, when executed, runs self and passes
        its result to f, then runs the resulting Cont.
        """
        return Cont(lambda k: self.run(lambda a: f(a).run(k)))

    def __call__(self, k: Callable[[A], R] = lambda x: x) -> R:  # type: ignore[assignment]
        """Execute the chain with the given (or identity) continuation."""
        return self.run(k)


# --- Workflow Steps ---

User = dict[str, str | int]


def validate_email(email: str) -> Cont[str, str]:
    """Validate email format. Short-circuits if invalid."""

    def logic(k: Callable[[str], str]) -> str:
        if "@" not in email:
            return f"Error: '{email}' is not a valid email address."
        return k(email)

    return Cont(logic)


def check_db_for_user(email: str) -> Cont[str, User]:
    """Check if user exists in database. Short-circuits if found."""

    def logic(k: Callable[[User], str]) -> str:
        if email == "admin@test.com":
            return "Error: User already exists in database."
        return k({"email": email, "id": 42})

    return Cont(logic)


def register_user(user: User) -> Cont[str, str]:
    """Finalise user registration."""

    def logic(k: Callable[[str], str]) -> str:
        return k(f"Success: Account created for {user['email']}")

    return Cont(logic)


# --- Tests ---


def test_pure_and_call() -> None:
    assert Cont.pure(42)() == 42


def test_bind_chains() -> None:
    # fmt: off
    assert (
        Cont.pure(5)
        << (lambda x: Cont.pure(x + 1))
        << (lambda x: Cont.pure(x * 2))
    )() == 12
    # fmt: on


def test_bind_associativity() -> None:
    def f(x: int) -> Cont[int, int]:
        return Cont.pure(x + 1)

    def g(x: int) -> Cont[int, int]:
        return Cont.pure(x * 2)

    m = Cont.pure(5)
    assert ((m << f) << g)() == (m << (lambda x: f(x) << g))()


def test_workflow_success() -> None:
    # fmt: off
    assert (
        validate_email("new@example.com")
        << check_db_for_user
        << register_user
    )() == "Success: Account created for new@example.com"
    # fmt: on


def test_short_circuit_invalid_email() -> None:
    # fmt: off
    assert (
        validate_email("invalid")
        << check_db_for_user
        << register_user
    )() == "Error: 'invalid' is not a valid email address."
    # fmt: on


def test_short_circuit_existing_user() -> None:
    # fmt: off
    assert (
        validate_email("admin@test.com")
        << check_db_for_user
        << register_user
    )() == "Error: User already exists in database."
    # fmt: on


def test_custom_continuation() -> None:
    assert Cont.pure(10)(lambda x: x * 3) == 30


# --- CLI ---

cli = click.Group()


@cli.command("test")
def run_tests() -> None:
    """Run pytest on this module."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "-p", "no:cacheprovider"],
        cwd=Path(__file__).parent,
    )
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    cli()
