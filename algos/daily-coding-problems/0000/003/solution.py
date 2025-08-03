"""
Problem
-------

Good morning! Here's your coding interview problem for today.

This problem was asked by Quora.

Given a string, find the palindrome that can be made by inserting the fewest
number of characters as possible anywhere in the word. If there is more than one
palindrome of minimum length that can be made, return the lexicographically
earliest one (the first one alphabetically).

For example, given the string "race", you should return "ecarace", since we can
add three letters to it (which is the smallest amount to make a palindrome).
There are seven other palindromes that can be made from "race" by adding three
letters, but "ecarace" comes first alphabetically.

As another example, given the string "google", you should return "elgoogle".
"""

from __future__ import annotations

import logging
from typing import Tuple

from .utils import Arguments, configure_logging

logger = logging.getLogger(__name__)


def main(args: Arguments):
    """
    Returns the first palindrome alphabetically which
    can be formed by adding the fewest letters to a word.
    """

    logger.debug(f"Received arguments: {args}")

    word = args.word

    assert len(word) > 0, "Word must have at least one letter!"

    palindromes = get_all_palindromes(word)

    sorted_palindromes = sort_palindromes_by_length_and_alphabetically(palindromes)

    return sorted_palindromes[0]


def get_all_palindromes(word: str) -> Tuple[str, ...]:
    """
    Returns all the palindromes which can be formed from a word.
    """


def sort_palindromes_by_length_and_alphabetically(
    palindromes: Tuple[str, ...],
) -> Tuple[str, ...]:
    """ """


if __name__ == "__main__":
    args = Arguments.parse_arguments()

    configure_logging(args)

    palindrome = main(args)

    print(palindrome)
