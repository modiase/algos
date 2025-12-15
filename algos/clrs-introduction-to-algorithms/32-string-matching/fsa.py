#!/usr/bin/env python3
"""FSA-based pattern matcher using functional programming and immutable types."""

import sys
from typing import NamedTuple, Callable, Optional, Iterator, NoReturn


class FSA(NamedTuple):
    states: frozenset[int]
    alphabet: frozenset[str]
    initial_state: int
    accept_states: frozenset[int]
    transition_fn: Callable[[int, str], Optional[int]]


def make_transition_fn(pattern: str) -> Callable[[int, str], Optional[int]]:
    """
    Higher-order function generating transition function for pattern matching.
    State i represents having matched the first i characters of the pattern.
    """
    pattern_tuple = tuple(pattern)
    pattern_len = len(pattern_tuple)

    transition_table = tuple(
        tuple(
            state + 1 if state < pattern_len and pattern_tuple[state] == char else None
            for char in pattern_tuple
        )
        for state in range(pattern_len + 1)
    )

    char_to_idx = {char: idx for idx, char in enumerate(pattern_tuple)}

    def transition(state: int, char: str) -> Optional[int]:
        if char not in char_to_idx:
            return None
        if state < len(transition_table):
            return transition_table[state][char_to_idx[char]]
        return None

    return transition


def build_fsa(pattern: str) -> FSA:
    pattern_len = len(pattern)
    return FSA(
        frozenset(range(pattern_len + 1)),
        frozenset(pattern),
        0,
        frozenset([pattern_len]),
        make_transition_fn(pattern),
    )


def try_transition_and_check(
    fsa: FSA, state: int, char: str, pos: int
) -> tuple[Optional[int], Optional[int]]:
    """Returns (next_state, match_position) where match_position is set if accept state reached."""
    next_state = fsa.transition_fn(state, char)
    if next_state is not None and next_state in fsa.accept_states:
        return fsa.initial_state, pos + 1
    return next_state, None


def find_matches(fsa: FSA, text: str) -> Iterator[int]:
    """Yield positions where pattern matches end (exclusive)."""
    state = fsa.initial_state

    for pos, char in enumerate(text):
        next_state, match_pos = try_transition_and_check(fsa, state, char, pos)

        if next_state is not None:
            state = next_state
            if match_pos is not None:
                yield match_pos
        else:
            new_state, match_pos = try_transition_and_check(
                fsa, fsa.initial_state, char, pos
            )
            state = new_state if new_state is not None else fsa.initial_state
            if match_pos is not None:
                yield match_pos


def read_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def error_exit(message: str) -> NoReturn:
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <pattern> <file_path>", file=sys.stderr)
        sys.exit(1)

    pattern = sys.argv[1]
    file_path = sys.argv[2]

    if not pattern:
        error_exit("Pattern cannot be empty")

    try:
        text = read_file(file_path)
    except FileNotFoundError:
        error_exit(f"File '{file_path}' not found")
    except IOError as e:
        error_exit(f"Reading file: {e}")

    fsa = build_fsa(pattern)
    matches = tuple(find_matches(fsa, text))

    print(f"Pattern: '{pattern}'")
    print(f"File: {file_path}")
    print(f"Matches found: {len(matches)}")

    if matches:
        print(f"\nMatch positions (end index): {matches}")


if __name__ == "__main__":
    main()
