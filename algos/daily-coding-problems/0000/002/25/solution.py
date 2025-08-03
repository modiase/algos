from __future__ import annotations

import logging
from collections import deque
from typing import Deque, List, NamedTuple, Optional, Union


class Attempt(NamedTuple):
    matched_string: str
    remaining_string: str
    remaining_tokens: List[Token]


class DecisionPoint(NamedTuple):
    greedy: Attempt
    non_greedy: Attempt


class Token(NamedTuple):
    character: str
    is_starred: bool = False


def tokenize(pattern: str) -> List[Token]:
    tokens = []
    while pattern:
        if len(pattern) == 1:
            tokens.append(Token(character=pattern))
            pattern = ""
        else:
            head = pattern[0]
            pattern = pattern[1:]
            if pattern[0] == "*":
                pattern = pattern[1:]
                tokens.append(Token(character=head, is_starred=True))
            else:
                tokens.append(Token(character=head))
    return tokens


def advance_attempt_starred(attempt: Attempt, greedy: bool = False) -> Attempt:
    next_token = attempt.remaining_tokens[0]
    next_token_char = next_token.character

    new_matched_string = attempt.matched_string + next_token_char
    if greedy:
        new_remaining_tokens = attempt.remaining_tokens
        new_remaining_string = attempt.remaining_string[1:]
    else:
        new_remaining_tokens = attempt.remaining_tokens[1:]
        new_remaining_string = attempt.remaining_string

    return Attempt(
        matched_string=new_matched_string,
        remaining_tokens=new_remaining_tokens,
        remaining_string=new_remaining_string,
    )


def advance_attempt(attempt: Attempt, greedy: bool = False) -> Attempt:
    next_token = attempt.remaining_tokens[0]
    next_token_char = next_token.character

    new_matched_string = attempt.matched_string + next_token_char
    new_remaining_tokens = attempt.remaining_tokens[1:]
    new_remaining_string = attempt.remaining_string[1:]

    return Attempt(
        matched_string=new_matched_string,
        remaining_tokens=new_remaining_tokens,
        remaining_string=new_remaining_string,
    )


def take_next_greedy(attempt: Attempt) -> Attempt:
    return advance_attempt_starred(attempt, greedy=True)


def take_next_non_greedy(attempt: Attempt, starred: bool = False) -> Attempt:
    if starred:
        return advance_attempt_starred(attempt)
    return advance_attempt(attempt)


def take_next(attempt: Attempt) -> Optional[Union[DecisionPoint, Attempt]]:
    """Returns None if no next match is possible; otherwise, returns either
    a single progression of the attempt — i.e., a single character moved into
    the matched field from the remaining field — or, in the case of a wildcard,
    both the greedy and non-greedy next possibilities."""

    if len(attempt.remaining_string) == 0 or len(attempt.remaining_tokens) == 0:
        return None
    next_token_to_match_in_pattern = attempt.remaining_tokens[0]
    next_char_to_match_in_string = attempt.remaining_string[0]

    if next_token_to_match_in_pattern.is_starred:
        return DecisionPoint(
            greedy=take_next_greedy(attempt),
            non_greedy=take_next_non_greedy(attempt, starred=True),
        )

    if (
        next_token_to_match_in_pattern.character == next_char_to_match_in_string
        or next_token_to_match_in_pattern.character == "."
    ):
        return take_next_non_greedy(attempt)


def main_loop(queue: Deque[Attempt]) -> bool:
    logging.debug(queue)
    logging.info(len(queue))

    attempt = queue.popleft()
    logging.info(attempt)

    if attempt.remaining_tokens == [] and attempt.remaining_string == "":
        return True

    next_step = take_next(attempt)

    if isinstance(next_step, DecisionPoint):
        queue.appendleft(next_step.non_greedy)
        queue.appendleft(next_step.greedy)
    elif next_step:
        queue.appendleft(next_step)
    return False


def main(pattern: str, string: str) -> bool:
    q = deque()
    q.append(
        Attempt(
            matched_string="",
            remaining_string=string,
            remaining_tokens=tokenize(pattern),
        )
    )
    match_found = False
    while len(q) != 0 and not match_found:
        match_found = main_loop(q)

    logging.info(match_found)
    return match_found


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(funcName)s"
    "- %(lineno)d : %(message)s",
)

if __name__ == "__main__":
    main(".*at.*rq", "chatsdatfrzafafrq")
