"""
Utility functions for solution to problem 33.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass


def configure_logging(args: Arguments):
    """
    Basic configuration for logging.
    """
    level = logging.INFO
    if args.debug:
        level = logging.DEBUG

    logging.basicConfig(level=level,
                        format='[{levelname} - {asctime} - {name}'
                        '- {funcName} - {lineno}]: {message}',
                        style='{')

@dataclass(frozen=True)
class Arguments:
    """
    Received arguments to run script.
    """
    word: str
    debug: bool = False

    @classmethod
    def parse_arguments(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument('word', help='The word from which a palindrome will'
                            'be created')
        parser.add_argument('--debug', '-d', action='store_true',
                            default=False, help='Enable debug logging.')

        args = parser.parse_args()
        arguments = vars(args)

        return cls(**arguments)
