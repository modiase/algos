from collections import namedtuple
from typing import List, Optional, Tuple

Token = namedtuple("Token", ("char", "run_length"))


def main(input_str: str) -> str:
    tokens = tokenise(None, 0, input_str, [])

    return "".join([f"{token.run_length}{token.char}" for token in tokens])


def tokenise(
    current_char: Optional[str],
    current_run_length: int,
    input_str: str,
    acc: List[Token],
) -> List[Token]:
    if not current_char:
        head, tail = take_first_char(input_str)
        return tokenise(head, 1, tail, acc)

    if input_str == "":
        if current_char:
            acc.append(Token(char=current_char, run_length=current_run_length))

        return acc

    head, tail = take_first_char(input_str)

    if head == current_char:
        return tokenise(head, current_run_length + 1, tail, acc)

    acc = acc + [(Token(current_char, current_run_length))]
    return tokenise(head, 1, tail, acc)


def take_first_char(input_str: str) -> Tuple[str, str]:
    head, *tail = input_str
    tail = "".join(tail)
    return head, tail
