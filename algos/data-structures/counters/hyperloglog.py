#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.loguru -p python313Packages.pytest -p python313Packages.click
from __future__ import annotations

from collections.abc import Iterable
from hashlib import sha256
from itertools import takewhile
from math import log
from typing import Any, ClassVar

import click
import pytest
from loguru import logger


class HyperLogLog:
    ALPHA: ClassVar = 0.673

    def __init__(self) -> None:
        self._b = 4
        self._n_buckets = 2**self._b
        self._buckets = [0] * self._n_buckets

    @staticmethod
    def ilen(it: Iterable[Any]) -> int:
        return sum(1 for _ in it)

    def add(self, val: int) -> None:
        bits = f"{int(sha256(str(val).encode()).hexdigest(), base=16):0256b}"
        bucket_idx = int(bits[: self._b], base=2)
        self._buckets[bucket_idx] = max(
            self.ilen(takewhile(lambda x: x == "0", bits[self._b :])) + 1,
            self._buckets[bucket_idx],
        )

    def estimate(self) -> float:
        raw_estimate = (
            self.ALPHA
            * self._n_buckets**2
            / sum(2 ** (-bucket) for bucket in self._buckets)
        )

        # Small range correction: When cardinality is low (< 2.5 * n_buckets), many buckets
        # remain empty (value 0). The harmonic mean formula assumes all buckets have seen
        # at least one element, causing systematic underestimation. Linear counting is more
        # accurate here: if V buckets are empty out of m total, the expected cardinality is
        # -m * ln(V/m) based on the coupon collector's problem. We're essentially asking:
        # "how many items must we add before V buckets remain untouched?"
        if raw_estimate <= 2.5 * self._n_buckets and self._buckets.count(0) > 0:
            return self._n_buckets * log(self._n_buckets / self._buckets.count(0))

        # Large range correction: When cardinality approaches 2^32 (the hash space size),
        # different elements start mapping to the same hash values (birthday paradox).
        # The raw estimate assumes no collisions, but with n elements in a space of size 2^32,
        # the probability of at least one collision grows. This formula inverts the expected
        # number of distinct hash values: if E = 2^32 * (1 - e^(-n/2^32)), solving for n gives
        # n = -2^32 * ln(1 - E/2^32). This corrects for the "missing" elements that collided.
        if raw_estimate > (1 << 32) / 30:
            return -(1 << 32) * log(1 - raw_estimate / (1 << 32))

        return raw_estimate


@pytest.mark.parametrize(
    ("n_items", "max_error_pct"),
    [
        (100, 50),
        (1000, 30),
        (10000, 30),
        (100000, 15),
    ],
)
def test_hyperloglog_estimate(n_items: int, max_error_pct: float) -> None:
    hll = HyperLogLog()
    for i in range(n_items):
        hll.add(i)
    error_pct = abs(hll.estimate() - n_items) / n_items * 100
    assert error_pct < max_error_pct, f"Error {error_pct:.2f}% exceeds {max_error_pct}%"


def test_hyperloglog_duplicates() -> None:
    hll = HyperLogLog()
    for _ in range(1000):
        hll.add(42)
    assert hll.estimate() < 10


cli = click.Group()


@cli.command()
@click.argument("n_items", type=int, default=100000)
def demo(n_items: int) -> None:
    """Run HyperLogLog demo with N_ITEMS distinct values."""
    hll = HyperLogLog()
    for i in range(n_items):
        hll.add(i)
    estimate = hll.estimate()
    logger.info(f"Estimated cardinality: {estimate:.0f}")
    logger.info(f"Actual cardinality: {n_items}")
    logger.debug(f"Error: {abs(estimate - n_items) / n_items * 100:.2f}%")


@cli.command("test")
def run_tests() -> None:
    """Run tests."""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    cli()
