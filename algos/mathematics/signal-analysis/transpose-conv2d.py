#!/usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.pytest -p python313Packages.loguru
"""2D Transpose Convolution Implementation"""

import sys
from collections.abc import Sequence
from dataclasses import dataclass

import pytest


def transpose_conv2d(
    x: Sequence[Sequence[float]],
    h: Sequence[Sequence[float]],
    padding: tuple[int, int],
    stride: tuple[int, int],
) -> Sequence[Sequence[float]]:
    H_in, W_in = len(x), len(x[0])
    K_h, K_w = len(h), len(h[0])
    stride_h, stride_w = stride
    pad_h, pad_w = padding

    H_full = H_in + (K_h - 1) * stride_h
    W_full = W_in + (K_w - 1) * stride_w
    result = [
        [0.0 for _ in range(W_full - 2 * pad_w)] for _ in range(H_full - 2 * pad_h)
    ]

    for i in range(H_in):
        for j in range(W_in):
            for k_h in range(K_h):
                for k_w in range(K_w):
                    if (
                        (i + k_h * stride_h) >= pad_h
                        and (i + k_h * stride_h) < H_full - pad_h
                        and (j + k_w * stride_w) >= pad_w
                        and (j + k_w * stride_w) < W_full - pad_w
                    ):
                        result[(i + k_h * stride_h) - pad_h][
                            (j + k_w * stride_w) - pad_w
                        ] += x[i][j] * h[k_h][k_w]

    return result


@dataclass(frozen=True, kw_only=True)
class TConv2dTestCase:
    x: Sequence[Sequence[float]]
    h: Sequence[Sequence[float]]
    padding: tuple[int, int]
    stride: tuple[int, int]
    expected: Sequence[Sequence[float]]

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
        TConv2dTestCase(
            x=[[1, 2], [3, 4]],
            h=[[1, 1], [1, 1]],
            padding=(0, 0),
            stride=(1, 1),
            expected=[[1, 3, 2], [4, 10, 6], [3, 7, 4]],
        ),
        TConv2dTestCase(
            x=[[1, 2], [3, 4]],
            h=[[1, 1], [1, 1]],
            padding=(1, 1),
            stride=(1, 1),
            expected=[[10]],
        ),
        TConv2dTestCase(
            x=[[1, 2], [3, 4]],
            h=[[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            padding=(0, 0),
            stride=(1, 1),
            expected=[[1, 3, 3, 2], [4, 10, 10, 6], [4, 10, 10, 6], [3, 7, 7, 4]],
        ),
        TConv2dTestCase(
            x=[[1, 2], [3, 4]],
            h=[[1, 1], [1, 1]],
            padding=(0, 0),
            stride=(2, 2),
            expected=[[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2], [3, 4, 3, 4]],
        ),
        TConv2dTestCase(
            x=[[5]],
            h=[[1, 2], [3, 4]],
            padding=(0, 0),
            stride=(1, 1),
            expected=[[5, 10], [15, 20]],
        ),
        TConv2dTestCase(
            x=[[5]], h=[[1, 2], [3, 4]], padding=(1, 1), stride=(1, 1), expected=[]
        ),
    ],
    ids=lambda test_case: f"x{len(test_case.x)}x{len(test_case.x[0])}_h{len(test_case.h)}x{len(test_case.h[0])}_p{test_case.padding}_s{test_case.stride}",
)
def test_transpose_conv2d(test_case: TConv2dTestCase):
    result = transpose_conv2d(**test_case.as_dict())
    assert result == test_case.expected


if __name__ == "__main__":
    pytest.main([__file__, *sys.argv[1:]])
