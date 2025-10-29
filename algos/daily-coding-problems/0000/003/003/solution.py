"""Solution to Daily Coding Problem 32:

Problem
=======
Good morning! Here's your coding interview problem for today.

This problem was asked by Jane Street.

Suppose you are given a table of currency exchange rates,
represented as a 2D array. Determine whether there is a possible arbitrage:
that is, whether there is some sequence of trades you can make, starting with
some amount A of any currency, so that you can end up with some amount greater
than A of that currency.

There are no transaction costs and you can trade fractional quantities.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional


def main(args: Arguments):
    conversion_rates = args.rates
    for start_currency in conversion_rates.currencies:
        detect_arbitrage(conversion_rates, start_currency)


def detect_arbitrage(
    conversion_rates: ConversionRates, start_currency: str
) -> Optional[Tuple[str, ...]]:
    """Loops through all possible trades to attempt to detect an arbitratge
    opportunity starting from a start currency returning a sequence of currency
    codes which corresponds to the ordered sequence of currencies which should
    be purchased, starting and ending with the start currency, in order to
    achieve a profit. If no such opportunity is found, None is returned."""
    pass


class ConversionRates:
    """Holds the input conversion rates for the currencies for which an arbitrage
    opportunity is being detected.
    """

    def __init__(self, path: Path):
        self._path = path
        self._dict = json.loads(self._path.read_text())
        self._currencies = self._dict.keys()

    @classmethod
    def from_argument(cls, path_argument: str):
        path = Path(path_argument)
        if not path.exists() or path.suffix != ".json":
            raise ValueError("Must supply a path to valid JSON.")
        return cls(path)

    @property
    def currencies(self):
        return self._currencies


@dataclass
class Arguments:
    rates: ConversionRates

    @classmethod
    def parse_args(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "currencies_file",
            help="The file containing exchange rates for each currency.",
            type=ConversionRates.from_argument,
        )
        return cls(**vars(parser.parse_args()))


if __name__ == "__main__":
    args = Arguments.parse_args()

    main(args)
