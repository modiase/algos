#!/usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.pytest -p python313Packages.loguru
""" """

import sys
from collections.abc import Sequence
from dataclasses import dataclass

import pytest


def transpose_conv1d(
    x: Sequence[float],
    h: Sequence[float],
    padding: int,
    stride: int,
) -> Sequence[float]:
    D, K = len(x), len(h)
    full_output_size = D + (K - 1) * stride
    result = [0.0] * (full_output_size - 2 * padding)

    for i in range(D):
        for k in range(K):
            if (i + k * stride) >= padding and (
                i + k * stride
            ) < full_output_size - padding:
                result[(i + k * stride) - padding] += x[i] * h[k]

    return result


@dataclass(frozen=True, kw_only=True)
class TConv1dTestCase:
    x: Sequence[float]
    h: Sequence[float]
    padding: int
    stride: int
    expected: Sequence[float]

    def as_dict(self):
        return {
            "x": self.x,
            "h": self.h,
            "padding": self.padding,
            "stride": self.stride,
        }


@pytest.mark.parametrize(
    "test_case",
    [
        TConv1dTestCase(x=[1, 2], h=[1, 2], padding=0, stride=1, expected=[1, 4, 4]),
        TConv1dTestCase(
            x=[1, 2], h=[1, 1, 1], padding=0, stride=1, expected=[1, 3, 3, 2]
        ),
        TConv1dTestCase(
            x=[1, 2, 3], h=[1, 2, 3], padding=0, stride=1, expected=[1, 4, 10, 12, 9]
        ),
        TConv1dTestCase(
            x=[1, 2, 3], h=[1, 2, 3], padding=1, stride=1, expected=[4, 10, 12]
        ),
        TConv1dTestCase(x=[1, 2, 3], h=[1, 2, 3], padding=2, stride=1, expected=[10]),
        TConv1dTestCase(x=[1, 1], h=[1, 1], padding=0, stride=2, expected=[1, 1, 1, 1]),
        TConv1dTestCase(
            x=[1, 1], h=[1, 1], padding=0, stride=3, expected=[1, 1, 0, 1, 1]
        ),
        TConv1dTestCase(x=[1, 1], h=[1, 1], padding=1, stride=3, expected=[1, 0, 1]),
        TConv1dTestCase(
            x=[1, 2, 3], h=[1, 1], padding=0, stride=2, expected=[1, 2, 4, 2, 3]
        ),
        TConv1dTestCase(
            x=[1, 2, 3], h=[1, 1], padding=0, stride=3, expected=[1, 2, 3, 1, 2, 3]
        ),
    ],
    ids=lambda test_case: f"{test_case.x=}, {test_case.h=}, {test_case.padding=}, {test_case.stride=}",
)
def test_reverse_conv1d(test_case: TConv1dTestCase):
    assert transpose_conv1d(**test_case.as_dict()) == test_case.expected


if __name__ == "__main__":
    pytest.main([__file__, *sys.argv[1:]])
