"""
Suppose we have an optimal prefix code on a set C = { 0, 1, ..., n} of
characters and we wish to transmit this code using as few bits as possible. Show
how to represent any optimal prefix code on C using only 2n-1 + n ceil(lg n) bits.
(Hint: Use 2n-1 bits to specify the structure of the tree, as discovered by a
walk of the tree.)
"""

from __future__ import annotations

import heapq
from collections import Counter, deque
from collections.abc import Mapping, Sequence
from itertools import chain
from pathlib import Path
from typing import overload

from more_itertools import ilen, one, with_iter


class HuffmanNode:
    @classmethod
    def exterior(cls, *, symbol: str, frequency: int) -> HuffmanNode:
        return cls(frequency=frequency, symbol=symbol)

    @overload
    @classmethod
    def interior(
        cls, *, left: HuffmanNode, right: HuffmanNode | None = None
    ) -> HuffmanNode: ...

    @overload
    @classmethod
    def interior(cls, *, left: None = None, right: HuffmanNode) -> HuffmanNode: ...

    @overload
    @classmethod
    def interior(cls, *, left: HuffmanNode, right: HuffmanNode) -> HuffmanNode: ...

    @classmethod
    def interior(
        cls, *, left: HuffmanNode | None = None, right: HuffmanNode | None = None
    ) -> HuffmanNode:
        if left is None or right is None:
            raise ValueError("Interior nodes must have both a left and right child.")
        return cls(frequency=left.frequency + right.frequency, left=left, right=right)

    def __init__(
        self,
        *,
        frequency: int | None = None,
        symbol: str | None = None,
        left: HuffmanNode | None = None,
        right: HuffmanNode | None = None,
    ):
        self.symbol = symbol
        self._frequency = frequency
        self.left = left
        self.right = right

    @property
    def frequency(self) -> int:
        if self._frequency is None:
            return self.left.frequency + self.right.frequency
        return self._frequency

    def __lt__(self, other: HuffmanNode) -> bool:
        return self.frequency < other.frequency

    def __eq__(self, other: HuffmanNode) -> bool:
        return self.frequency == other.frequency

    def dump_representation(self) -> str:
        stack = [self]
        output = []
        while stack:
            node = stack.pop()
            if node.left is None and node.right is None:
                output.append("1")
                continue
            output.append("0")
            if node.right is not None:
                stack.append(node.right)
            if node.left is not None:
                stack.append(node.left)
        return "".join(output)

    def dump_symbols_bfs(self) -> Sequence[str]:
        output = []
        queue = deque([self])
        while queue:
            node = queue.popleft()
            if node.symbol:
                output.append(node.symbol)
            if node.left is not None:
                queue.append(node.left)
            if node.right is not None:
                queue.append(node.right)
        return output

    def generate_code(self) -> dict[str, str]:
        """
        Generate Huffman codes for all symbols in the tree.
        Uses the convention: left = 0, right = 1

        Returns:
            A dictionary mapping symbols to their binary codes.
        """
        if self.left is None and self.right is None:
            # Special case: single node tree
            return {self.symbol: "0"} if self.symbol is not None else {}

        codes = {}
        # Use a stack to track nodes and their current code paths
        # Each stack item is a tuple: (node, current_code)
        stack = [(self, "")]

        while stack:
            node, code = stack.pop()

            # If this is a leaf node with a symbol
            if node.left is None and node.right is None and node.symbol is not None:
                codes[node.symbol] = code
                continue

            # Push right child first (so left gets processed first when popped)
            if node.right is not None:
                stack.append((node.right, code + "1"))

            # Push left child
            if node.left is not None:
                stack.append((node.left, code + "0"))

        return codes

    @classmethod
    def from_representation(
        cls, representation: str, symbols: Sequence[str]
    ) -> HuffmanNode:
        if len(representation) == 0:
            raise ValueError("Representation must be at least one bit long.")
        if representation[0] != "0":
            raise ValueError("Representation must start with a 0.")
        if len(representation) % 2 != 1:
            # A valid representation of a full binary tree with n leaves must
            # have 2n-1 bits.
            # Assume n > 1.
            # n = 2 => 2*2-1 = 3 nodes 011
            # k leaves => 2k-1 . Adding one interior node adds two nodes.
            # 2k - 1 + 2 = 2k + 1 = 2 (k + 1) - 1
            # True for n = 1 and n = k + 1 if true for n = k => true for all n.
            raise ValueError("Representation must have an odd number of bits.")

        # Count the number of leaf nodes (1s) in the representation
        leaf_count = sum(1 for c in representation if c == "1")
        if leaf_count != len(symbols):
            raise ValueError(
                f"Expected {leaf_count} symbols for {leaf_count} leaves, got {len(symbols)}"
            )

        # The idea is to simulate DFS tree traversal the same way
        # dump_representation does. We'll use a single index to track our
        # position in the representation
        position = [0]

        def build_node():
            if position[0] >= len(representation):
                raise ValueError("Premature end of representation string")

            bit = representation[position[0]]
            position[0] += 1

            if bit == "1":
                # Leaf node
                return cls(frequency=0)
            elif bit == "0":
                # Interior node
                node = cls(frequency=0)
                node.left = build_node()
                node.right = build_node()
                return node
            else:
                raise ValueError(f"Invalid character in representation: {bit}")

        root = build_node()

        if position[0] != len(representation):
            raise ValueError(
                f"Not all of representation was consumed: {position[0]}/{len(representation)}"
            )

        symbol_idx = 0
        queue = deque([root])

        while queue and symbol_idx < len(symbols):
            node = queue.popleft()

            if node.left is None and node.right is None:
                node.symbol = symbols[symbol_idx]
                symbol_idx += 1

            if node.left is not None:
                queue.append(node.left)
            if node.right is not None:
                queue.append(node.right)

        if symbol_idx != len(symbols):
            raise ValueError(
                f"Not all symbols were assigned: {symbol_idx}/{len(symbols)}"
            )

        return root

    @classmethod
    def of(cls, char_to_freq: Mapping[str, int]) -> HuffmanNode:
        queue = [
            cls(frequency=freq, symbol=symbol) for symbol, freq in char_to_freq.items()
        ]
        heapq.heapify(queue)
        while len(queue) > 1:
            left = heapq.heappop(queue)
            right = heapq.heappop(queue)
            heapq.heappush(
                queue,
                HuffmanNode(
                    frequency=left.frequency + right.frequency,
                    left=left,
                    right=right,
                ),
            )
        return one(queue)


class Node:
    def __init__(
        self,
        *,
        symbol: str | None = None,
        left: Node | None = None,
        right: Node | None = None,
    ):
        self.left = left
        self.right = right
        self.symbol = symbol

    @classmethod
    def exterior(cls, *, symbol: str) -> Node:
        return cls(symbol=symbol)

    @classmethod
    def interior(cls, *, left: Node, right: Node) -> Node:
        return cls(left=left, right=right)

    def __repr__(self) -> str:
        return f"Node({f'symbol={self.symbol}' if self.symbol else ''}{f'left={repr(self.left)}' if self.left else ''}{f' right={repr(self.right)}' if self.right else ''})"


if __name__ == "__main__":

    def iter_file_chars():
        for char in chain.from_iterable(
            with_iter(open(Path(__file__).parent / "passage.txt"))
        ):
            yield char

    huffman_tree = HuffmanNode.of(Counter(iter_file_chars()))
    print("Tree representation:", huffman_tree.dump_representation())
    print("Symbols:", huffman_tree.dump_symbols_bfs())

    print("\nHuffman Codes:")
    codes = huffman_tree.generate_code()
    for symbol, code in sorted(codes.items()):
        print(f"{repr(symbol)}: {code}")

    total_bits = sum(
        len(code) * freq
        for symbol, code in codes.items()
        for char, freq in Counter(
            chain.from_iterable(with_iter(open(Path(__file__).parent / "passage.txt")))
        ).items()
        if symbol == char
    )

    og_total_bits = sum(len(c.encode("utf-8")) * 8 for c in iter_file_chars())
    print("\nTotal characters:", ilen(iter_file_chars()))
    print(f"Total bits required for encoding: {total_bits}")
    print(f"Uncompressed character bits: {og_total_bits}")
    print(f"Compression ratio: {og_total_bits / total_bits}")
