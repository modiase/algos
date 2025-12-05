#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.click -p python313Packages.graphviz -p python313Packages.numpy -p python313Packages.loguru -p python313Packages.more-itertools
from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from contextlib import ExitStack
from heapq import heapify, heappop, heappush
from itertools import chain
from pathlib import Path

import click
import graphviz
import numpy as np
from loguru import logger
from more_itertools import ilen


def it_file_chars(path: Path) -> Iterator[str]:
    yield from chain.from_iterable(
        line.strip() for line in ExitStack().enter_context(path.open())
    )


class HuffmanNode:
    def __init__(
        self,
        *,
        symbol: str | None = None,
        frequency: int,
    ):
        self.symbol = symbol
        self.frequency = frequency
        self.left: HuffmanNode | None = None
        self.right: HuffmanNode | None = None

    def __lt__(self, other: HuffmanNode) -> bool:
        return self.frequency < other.frequency


def visualize_huffman_tree(root: HuffmanNode) -> None:
    dot = graphviz.Digraph()
    dot.attr(rankdir="TB", size="24,20")

    def add_nodes(node: HuffmanNode | None) -> None:
        if node:
            node_id = str(id(node))
            label = str(node.frequency)
            if node.symbol:
                label = f"{node.symbol}\\n{label}"

            dot.node(
                node_id,
                label=label,
                shape="box",
                style="filled",
                fillcolor="lightblue",
                fontsize="10",
                fontweight="bold",
            )

            if node.left:
                left_id = str(id(node.left))
                dot.edge(node_id, left_id, label="0")
                add_nodes(node.left)

            if node.right:
                right_id = str(id(node.right))
                dot.edge(node_id, right_id, label="1")
                add_nodes(node.right)

    add_nodes(root)

    dot.render("huffman_tree", format="png", cleanup=True)
    dot.view()


@click.group()
def cli() -> None: ...


@cli.command()
def compress() -> None:
    input_path = Path(__file__).parent.joinpath("huffman_codes.txt")

    input_symbols = Counter(it_file_chars(input_path))
    nodes: list[HuffmanNode] = [
        HuffmanNode(symbol=symbol, frequency=frequency)
        for symbol, frequency in input_symbols.items()
    ]
    heapify(nodes)

    while len(nodes) > 1:
        left = heappop(nodes)
        right = heappop(nodes)
        merged = HuffmanNode(frequency=left.frequency + right.frequency)
        merged.left = left
        merged.right = right
        heappush(nodes, merged)

    stack: list[tuple[str, HuffmanNode]] = [("", nodes[0])]
    symbol_to_code: dict[str, str] = {}
    while stack:
        prefix, node = stack.pop()
        if node.symbol:
            symbol_to_code[node.symbol] = prefix
        if node.left:
            stack.append((prefix + "0", node.left))
        if node.right:
            stack.append((prefix + "1", node.right))

    output_seq = "".join(symbol_to_code[symbol] for symbol in it_file_chars(input_path))
    packed = np.packbits([int(c) for c in output_seq])
    og_len = ilen(it_file_chars(input_path))
    compressed_len = len(packed) + len(json.dumps(dict(symbol_to_code)))
    logger.info(f"Original length: {og_len}")
    logger.info(f"Compressed length: {compressed_len}")
    logger.info(f"Compression ratio: {og_len / compressed_len}")
    Path(__file__).parent.joinpath("huffman_codes_code.txt").write_text(
        json.dumps(dict(symbol_to_code))
    )
    Path(__file__).parent.joinpath("huffman_codes_coded.txt").write_bytes(packed)


@cli.command()
def decompress() -> None:
    input_path = Path(__file__).parent.joinpath("huffman_codes_coded.txt")
    packed = np.array(list(input_path.read_bytes()), dtype=np.uint8)
    code_path = Path(__file__).parent.joinpath("huffman_codes_code.txt")
    code_to_symbol = {v: k for k, v in json.loads(code_path.read_text()).items()}
    buffer = ""
    raw_output_seq: list[str] = []
    for bit in np.unpackbits(packed):
        buffer += str(bit)
        if buffer in code_to_symbol:
            raw_output_seq.append(code_to_symbol[buffer])
            buffer = ""
    print("".join(raw_output_seq))


@cli.command()
def visualize() -> None:
    input_path = Path(__file__).parent.joinpath("huffman_codes.txt")
    input_symbols = Counter(it_file_chars(input_path))
    nodes: list[HuffmanNode] = [
        HuffmanNode(symbol=symbol, frequency=frequency)
        for symbol, frequency in input_symbols.items()
    ]
    heapify(nodes)
    while len(nodes) > 1:
        left = heappop(nodes)
        right = heappop(nodes)
        merged = HuffmanNode(frequency=left.frequency + right.frequency)
        merged.left = left
        merged.right = right
        heappush(nodes, merged)
    visualize_huffman_tree(nodes[0])


if __name__ == "__main__":
    cli()
