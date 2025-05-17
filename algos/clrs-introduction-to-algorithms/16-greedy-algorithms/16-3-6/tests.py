import importlib
from collections import Counter

import pytest

module = importlib.import_module("16-3-6")
HuffmanNode = module.HuffmanNode


def terminal_node() -> HuffmanNode:
    return HuffmanNode.interior(
        left=HuffmanNode.exterior(symbol="a", frequency=1),
        right=HuffmanNode.exterior(symbol="b", frequency=2),
    )


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (
            HuffmanNode.exterior(symbol="a", frequency=1),
            HuffmanNode.exterior(symbol="b", frequency=2),
            "011",
        ),
        (
            terminal_node(),
            HuffmanNode.exterior(symbol="c", frequency=3),
            "00111",
        ),
        (
            HuffmanNode.exterior(symbol="c", frequency=3),
            terminal_node(),
            "01011",
        ),
        (
            terminal_node(),
            terminal_node(),
            "0011011",
        ),
    ],
)
def test_dump_representation(left: HuffmanNode, right: HuffmanNode, expected: str):
    assert (
        HuffmanNode.interior(left=left, right=right).dump_representation() == expected
    )


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "aabc",
            HuffmanNode.interior(
                left=HuffmanNode.interior(
                    left=HuffmanNode.exterior(symbol="a", frequency=2),
                    right=HuffmanNode.exterior(symbol="b", frequency=1),
                ),
                right=HuffmanNode.exterior(symbol="c", frequency=1),
            ),
        ),
        (
            "aaabbccd",
            HuffmanNode.interior(
                left=HuffmanNode.interior(
                    left=HuffmanNode.exterior(symbol="a", frequency=3),
                    right=HuffmanNode.exterior(symbol="b", frequency=2),
                ),
                right=HuffmanNode.interior(
                    left=HuffmanNode.exterior(symbol="c", frequency=2),
                    right=HuffmanNode.exterior(symbol="d", frequency=1),
                ),
            ),
        ),
    ],
)
def test_of(text: str, expected: HuffmanNode):
    assert HuffmanNode.of(Counter(text)) == expected


def test_reconstruction():
    root = terminal_node()
    assert (
        root.from_representation(
            root.dump_representation(), root.dump_symbols_bfs()
        ).dump_representation()
        == root.dump_representation()
    )


def test_generate_code():
    root = terminal_node()
    assert root.generate_code() == {"a": "0", "b": "1"}
