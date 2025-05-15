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
from itertools import batched
from typing import assert_never

from more_itertools import one


class HuffmanNode:
    @classmethod
    def exterior(cls, *, symbol: str, frequency: int) -> HuffmanNode:
        return cls(frequency=frequency, symbol=symbol)

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
        frequency: int,
        symbol: str | None = None,
        left: HuffmanNode | None = None,
        right: HuffmanNode | None = None,
    ):
        if (left is None and right is None) == (symbol is None):
            raise ValueError(
                "Exterior nodes must have a symbol. Interior nodes must not have a symbol"
            )

        self.symbol = symbol
        self.frequency = frequency
        self.left = left
        self.right = right

    def __lt__(self, other: HuffmanNode) -> bool:
        return self.frequency < other.frequency

    def __eq__(self, other: HuffmanNode) -> bool:
        return self.frequency == other.frequency

    def dump_codes(self) -> Mapping[str, str]:
        codes = {}
        stack = [(self, "")]
        while stack:
            node, code = stack.pop()
            if node.symbol:
                codes[node.symbol] = code
            if node.right is not None:
                stack.append((node.right, code + "1"))
            if node.left is not None:
                stack.append((node.left, code + "0"))
        return codes

    def dump_representation(self) -> str:
        queue = deque([self])
        output = []
        while queue:
            node = queue.popleft()
            if node.left is None and node.right is None:
                output.append("1")
                continue
            output.append("0")
            if node.right is not None:
                queue.append(node.right)
            if node.left is not None:
                queue.append(node.left)
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


def huffman_codes(characters: Counter) -> Mapping[str, str]:
    queue = [
        HuffmanNode(frequency=freq, symbol=symbol)
        for symbol, freq in characters.items()
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

    @classmethod
    def from_representation(cls, representation: str, symbols: Sequence[str]) -> Node:
        print(representation)
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
        root = Node()
        stack = [root]
        symbol_queue = deque(symbols)
        # Build the interior nodes from the representation.
        for bit_pair in batched(representation[1:], 2):
            match bit_pair:
                case ("0", "0"):
                    stack[-1].right = Node()
                    stack.append(stack[-1].right)
                    stack[-1].left = Node()
                    stack.append(stack[-1].left)
                    stack.pop()
                case ("0", "1"):
                    stack[-1].left = Node()
                    stack.append(stack[-1].left)
                case ("1", "0"):
                    stack[-1].right = Node()
                    stack.append(stack[-1].right)
                case ("1", "1"):
                    stack.pop()
                case never:
                    assert_never(never)

        print(root)
        breakpoint()
        queue = deque([root])
        while queue:
            node = queue.popleft()
            if node.left is None:
                node.left = Node(symbol=symbol_queue.popleft())
            else:
                queue.append(node.left)
            if node.right is None:
                node.right = Node(symbol=symbol_queue.popleft())
            else:
                queue.append(node.right)
        if len(symbol_queue) != 0:
            raise ValueError("Unterminated symbol queue found.")
        return root


if __name__ == "__main__":
    t3 = HuffmanNode.interior(
        right=HuffmanNode.exterior(symbol="a", frequency=1),
        left=HuffmanNode.interior(
            left=HuffmanNode.exterior(symbol="b", frequency=2),
            right=HuffmanNode.exterior(symbol="c", frequency=3),
        ),
    )
    t4 = HuffmanNode.interior(
        left=t3, right=HuffmanNode.exterior(symbol="d", frequency=4)
    )

    print(Node.from_representation(t4.dump_representation(), t4.dump_symbols_bfs()))
    # huffman_tree = huffman_codes(
    #     Counter(
    #         chain.from_iterable(with_iter(open(Path(__file__).parent / "passage.txt")))
    #     )
    # )
    # print(huffman_tree.dump_representation())
