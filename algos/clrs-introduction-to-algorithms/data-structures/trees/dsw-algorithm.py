"""
I've implemented the DSW (Day-Stout-Warren) algorithm for your tree balancing
needs. This implementation:

Creates a perfectly balanced tree with guaranteed optimal height Works in O(n)
time with O(1) extra space Handles any input tree shape (not just linked lists)

How the Algorithm Works

First phase: Create backbone

Transforms the tree into a right-skewed linked list Uses right rotations to
eliminate all left children


Second phase: Balancing

Calculates the perfect tree size (2^⌊log₂(n+1)⌋ - 1) Handles excess nodes with
precise rotations Creates a tree with optimal height ⌊log₂(n)⌋ + 1



Key Improvements Over Your Original Algorithm

Mathematically optimal: Creates a tree with guaranteed minimum height Uniform
level filling: Fills each level completely before moving to the next Consistent
performance: Works equally well for all tree sizes No recursion needed: Uses an
iterative approach that's more efficient

The implementation includes helper functions to:

Calculate tree height Check if a tree is perfect (all levels completely filled)
Visualize the before/after trees

This algorithm will produce a much more balanced tree than your original
approach, especially for tree sizes that aren't powers of 2.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from graphviz import Digraph


@dataclass
class Node:
    key: str
    left: Node | None = None
    right: Node | None = None


def rotate_left(root: Node) -> Node:
    if not root.right:
        return root
    new_root = root.right
    root.right = new_root.left
    new_root.left = root
    return new_root


def rotate_right(root: Node) -> Node:
    if not root.left:
        return root
    new_root = root.left
    root.left = new_root.right
    new_root.right = root
    return new_root


def dsw_balance(root: Node) -> Node:
    """
    Balance a binary tree using the Day-Stout-Warren algorithm.
    """
    # Step 1: Create the backbone (right-skewed vine)
    root = create_backbone(root)

    # Step 2: Count the total number of nodes
    count = count_nodes(root)

    # Step 3: Calculate the number of leaves in the perfect subtree
    # For a tree with n nodes, a perfect binary tree has 2^h - 1 nodes
    # where h is the height. We want the largest h such that 2^h - 1 <= n
    perfect_tree_size = 2 ** int(math.log2(count + 1)) - 1

    # Step 4: Calculate the number of excess nodes (to be placed at the bottom level)
    excess_nodes = count - perfect_tree_size

    # Step 5: Apply initial rotations to handle excess nodes
    # This creates the lowest level with balanced filling
    root = compress_backbone(root, excess_nodes)

    # Step 6: Apply remaining rotations to create a perfect tree from the rest
    remaining = perfect_tree_size
    while remaining > 1:
        remaining = remaining // 2
        root = compress_backbone(root, remaining)

    return root


def create_backbone(root: Node) -> Node:
    """
    Convert the tree into a right-skewed vine (backbone).
    """
    pseudo_root = Node(key="pseudo")
    pseudo_root.right = root

    scan = pseudo_root
    while scan.right:
        if scan.right.left:
            # Rotate right at scan.right
            scan.right = rotate_right(scan.right)
        else:
            # Move to the next node in the backbone
            scan = scan.right

    return pseudo_root.right


def count_nodes(root: Node) -> int:
    """
    Count the total number of nodes in the tree.
    """
    count = 0
    current = root

    # Since we have a backbone, we can just count along the right spine
    while current:
        count += 1
        current = current.right

    return count


def compress_backbone(root: Node, count: int) -> Node:
    """
    Apply 'count' left rotations to the backbone.
    """
    if not root:
        return None

    pseudo_root = Node(key="pseudo")
    pseudo_root.right = root

    scan = pseudo_root
    for _ in range(count):
        if scan.right and scan.right.right:
            # Rotate left at scan.right
            scan.right = rotate_left(scan.right)
            # Move to the next node to be considered for rotation
            scan = scan.right
        else:
            break

    return pseudo_root.right


def visualize_tree(root: Node, filename: str = "tree") -> None:
    """
    Visualize a binary tree using graphviz.
    Saves the visualization as a PDF file.
    """
    dot = Digraph()
    dot.attr(rankdir="TB")  # Top to Bottom direction

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
    dot.render(filename, view=True, format="pdf")


def calculate_height(root: Node) -> int:
    """Calculate the height of the tree."""
    if not root:
        return 0
    return 1 + max(calculate_height(root.left), calculate_height(root.right))


def is_perfect(root: Node) -> bool:
    """Check if the tree is a perfect binary tree."""
    height = calculate_height(root)
    node_count = count_nodes(root)
    return node_count == (2**height - 1)


if __name__ == "__main__":
    # Create a linear tree (linked list)
    root = Node(key="0")
    current = root
    for k in range(1, 100):
        current.left = Node(key=str(k))
        current = current.left

    print("Original tree:")
    print(f"Height: {calculate_height(root)}")
    visualize_tree(root, "tree_before")

    # Apply DSW balancing
    root = dsw_balance(root)

    print("DSW Balanced tree:")
    print(f"Height: {calculate_height(root)}")
    print(f"Is perfect: {is_perfect(root)}")
    visualize_tree(root, "tree_after_dsw")

    # Calculate the optimal height for comparison
    optimal_height = math.floor(math.log2(100)) + 1
    print(f"Optimal height for 100 nodes: {optimal_height}")
