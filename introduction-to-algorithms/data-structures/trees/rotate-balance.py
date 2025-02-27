from __future__ import annotations

import math as m
from dataclasses import dataclass
from graphviz import Digraph

@dataclass
class Node:
    key: str
    left: Node | None = None
    right: Node | None = None


def ldepth(root: Node) -> int:
    """Calculate the depth of the left spine of the tree"""
    depth = 0
    current = root
    while current:
        depth += 1
        current = current.left
    return depth

def rdepth(root: Node) -> int:
    """Calculate the depth of the right spine of the tree"""
    depth = 0
    current = root
    while current:
        depth += 1
        current = current.right
    return depth

def lfold(root: Node) -> Node:
    depth = ldepth(root)
    for _ in range(m.ceil(depth / 2)-1):
        root = rotate_right(root)
    return root

def rfold(root: Node) -> Node:
    depth = rdepth(root)
    for _ in range(m.ceil(depth / 2)-1):
        root = rotate_left(root)
    return root


def balance(root: Node) -> Node:
    ld = ldepth(root)
    rd = rdepth(root)
    if ld > rd:
        root = lfold(root)
    elif rd > ld:
        root = rfold(root)
    if (left:=root.left):
        root.left = lfold(left)
        root.left = balance(root.left)
    if (right:=root.right):
        root.right = rfold(right)
        root.right = balance(root.right)
    return root


def rotate_left(root: Node) -> Node:
    if not root.right:
        return root
    tmp = (right:=root.right).left
    right.left = root
    root.right = tmp
    return right


def rotate_right(root: Node) -> Node:
    if not root.left:
        return root
    tmp = (left:=root.left).right
    left.right = root
    root.left = tmp
    return left


def visualize_tree(root: Node, filename: str = "tree") -> None:
    """
    Visualize a binary tree using graphviz.
    Saves the visualization as a PDF file.
    """
    dot = Digraph()
    dot.attr(rankdir='TB')  # Top to Bottom direction
    
    def add_nodes_edges(node: Node) -> None:
        if node is None:
            return
        
        # Add current node
        dot.node(str(node.key), str(node.key))
        
        # Add left child and edge
        if node.left:
            dot.node(str(node.left.key), str(node.left.key))
            dot.edge(str(node.key), str(node.left.key))
            add_nodes_edges(node.left)
        
        # Add right child and edge
        if node.right:
            dot.node(str(node.right.key), str(node.right.key))
            dot.edge(str(node.key), str(node.right.key))
            add_nodes_edges(node.right)
    
    add_nodes_edges(root)
    dot.render(filename, view=True, format='pdf')


if __name__ == '__main__':
    root = Node(key='0')
    current = root
    for k in range(1, 100):
        current.left = Node(key=str(k))
        current = current.left
    
    print("Original tree:")
    visualize_tree(root, "tree_before")
    
    root = balance(root)
    print("Balanced tree:")
    visualize_tree(root, "tree_after")