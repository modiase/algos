from collections import Counter
from heapq import heapify, heappop, heappush
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
from bidict import bidict


class HuffmanNode:
    def __init__(
        self,
        *,
        symbol: str | None = None,
        frequency: int,
    ):
        self.symbol = symbol
        self.frequency = frequency
        self.left = None
        self.right = None

    def __lt__(self, other: "HuffmanNode"):
        return self.frequency < other.frequency


def visualize_huffman_tree(root: HuffmanNode):
    G = nx.Graph()
    pos = {}
    labels = {}

    def add_nodes(node, x=0, y=0, layer=1):
        if node:
            node_id = id(node)
            G.add_node(node_id)
            pos[node_id] = (x, y)
            labels[node_id] = str(node.frequency)
            if node.symbol:
                labels[node_id] = f"{node.symbol}\n{labels[node_id]}"
            if node.left:
                left_id = id(node.left)
                G.add_edge(node_id, left_id)
                add_nodes(node.left, x - 5 / layer, y - 3, layer + 0.5)

            if node.right:
                right_id = id(node.right)
                G.add_edge(node_id, right_id)
                add_nodes(node.right, x + 5 / layer, y - 3, layer + 0.5)

    add_nodes(root)

    plt.figure(figsize=(20, 16))
    nx.draw(
        G,
        pos=pos,
        labels=labels,
        with_labels=True,
        node_color="lightblue",
        node_size=2000,
        font_size=8,
        font_weight="bold",
    )
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    input_seq = Path(__file__).parent.joinpath("huffman_codes.txt").read_text().split()
    input_symbols = Counter(input_seq)
    nodes = [
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

    stack = [("", nodes[0])]
    symbol_to_code = bidict()
    while stack:
        prefix, node = stack.pop()
        if node.symbol:
            symbol_to_code[node.symbol] = prefix.ljust(8, "0")
        if node.left:
            stack.append((prefix + "0", node.left))
        if node.right:
            stack.append((prefix + "1", node.right))

    output_seq = "".join(chr(int(symbol_to_code[symbol], 2)) for symbol in input_seq)
    print(len(" ".join(input_seq)))
    print(len(output_seq))
    Path(__file__).parent.joinpath("huffman_codes_coded.txt").write_text(output_seq)
