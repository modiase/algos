#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.pytest -p python313Packages.click
"""
Merkle Tree Implementation

A hash tree where leaf nodes contain hashes of data blocks and internal nodes
contain hashes of their children. Enables efficient verification of data
integrity and identification of differing blocks between two data sets.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import click
import pytest

HashFn = Callable[[bytes], bytes]


def sha256_hash(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


@dataclass(frozen=True)
class MerkleTree:
    """Merkle tree stored as levels for easy proof generation."""

    levels: tuple[tuple[bytes, ...], ...]
    num_leaves: int

    @property
    def root(self) -> bytes:
        return self.levels[-1][0]


@dataclass(frozen=True)
class MerkleProof:
    """
    Inclusion proof for a leaf in a Merkle tree.

    Contains the minimum information needed to verify membership without
    the full tree: the leaf's hash and the sibling hashes along the path
    to the root. The boolean indicates whether each sibling is on the left
    (True) or right (False), which determines concatenation order when
    recomputing parent hashes.

    Proof size is O(log n) regardless of tree size.
    """

    leaf_hash: bytes
    path: tuple[tuple[bytes, bool], ...]


def build_tree(
    data: Sequence[bytes], hash_fn: HashFn = sha256_hash
) -> MerkleTree | None:
    """
    Build a Merkle tree from a sequence of data blocks.

    Returns None if data is empty.
    Duplicates the last leaf if the count is odd at any level.
    """
    if not data:
        return None

    num_leaves = len(data)
    levels: list[tuple[bytes, ...]] = []

    current_level = tuple(hash_fn(block) for block in data)
    levels.append(current_level)

    while len(current_level) > 1:
        if len(current_level) % 2 == 1:
            current_level = current_level + (current_level[-1],)

        next_level = tuple(
            hash_fn(current_level[i] + current_level[i + 1])
            for i in range(0, len(current_level), 2)
        )
        levels.append(next_level)
        current_level = next_level

    return MerkleTree(levels=tuple(levels), num_leaves=num_leaves)


def generate_proof(tree: MerkleTree, leaf_index: int) -> MerkleProof | None:
    """
    Generate an inclusion proof for the leaf at the given index.

    Traverses from leaf to root, collecting sibling hashes at each level.
    The proof allows a verifier to recompute the root hash using only:
    - The leaf data
    - The sibling hashes along the path
    - The position (left/right) of each sibling

    This is O(log n) in both time and space.

    Returns None if the index is out of bounds.
    """
    if leaf_index < 0 or leaf_index >= tree.num_leaves:
        return None

    path: list[tuple[bytes, bool]] = []
    idx = leaf_index

    for level in tree.levels[:-1]:
        level_with_dup = level if len(level) % 2 == 0 else level + (level[-1],)

        if idx % 2 == 0:
            sibling_idx = idx + 1
            sibling_is_left = False
        else:
            sibling_idx = idx - 1
            sibling_is_left = True

        path.append((level_with_dup[sibling_idx], sibling_is_left))
        idx //= 2

    return MerkleProof(leaf_hash=tree.levels[0][leaf_index], path=tuple(path))


def verify_proof(
    root_hash: bytes,
    leaf_data: bytes,
    proof: MerkleProof,
    hash_fn: HashFn = sha256_hash,
) -> bool:
    """
    Verify that leaf_data is included in the tree with the given root hash.

    Recomputes the root by:
    1. Hashing the leaf data
    2. For each sibling in the proof path, concatenating hashes in the
       correct order (sibling left or right) and hashing the result
    3. Comparing the final hash to the expected root

    Returns True only if the recomputed root matches. Any tampering with
    the leaf data or proof will produce a different root hash.
    """
    current_hash = hash_fn(leaf_data)

    if current_hash != proof.leaf_hash:
        return False

    for sibling_hash, sibling_is_left in proof.path:
        if sibling_is_left:
            current_hash = hash_fn(sibling_hash + current_hash)
        else:
            current_hash = hash_fn(current_hash + sibling_hash)

    return current_hash == root_hash


def diff_trees(tree1: MerkleTree | None, tree2: MerkleTree | None) -> list[int]:
    """
    Find indices of differing leaf blocks between two trees.

    Efficiently prunes subtrees with matching hashes.
    Both trees must have the same number of leaves.
    """
    if tree1 is None and tree2 is None:
        return []
    if tree1 is None or tree2 is None:
        return list(
            range(
                max(tree1.num_leaves if tree1 else 0, tree2.num_leaves if tree2 else 0)
            )
        )
    if tree1.root == tree2.root:
        return []
    if tree1.num_leaves != tree2.num_leaves:
        raise ValueError("Trees must have the same number of leaves")

    differences: list[int] = []

    def find_diffs(level_idx: int, node_idx: int, start_leaf: int, span: int) -> None:
        if level_idx < 0:
            return

        level1 = tree1.levels[level_idx]
        level2 = tree2.levels[level_idx]

        level1_dup = level1 if len(level1) % 2 == 0 else level1 + (level1[-1],)
        level2_dup = level2 if len(level2) % 2 == 0 else level2 + (level2[-1],)

        if node_idx >= len(level1_dup):
            return

        if level1_dup[node_idx] == level2_dup[node_idx]:
            return

        if level_idx == 0:
            if start_leaf < tree1.num_leaves:
                differences.append(start_leaf)
            return

        half_span = span // 2
        find_diffs(level_idx - 1, node_idx * 2, start_leaf, half_span)
        find_diffs(level_idx - 1, node_idx * 2 + 1, start_leaf + half_span, half_span)

    total_span = 1 << (len(tree1.levels) - 1)
    find_diffs(len(tree1.levels) - 1, 0, 0, total_span)

    return differences


def tree_to_str(tree: MerkleTree) -> str:
    """Return a string representation of the tree with hex hashes."""
    lines = []
    for i, level in enumerate(reversed(tree.levels)):
        depth = len(tree.levels) - 1 - i
        indent = "  " * depth
        hashes = " ".join(h[:4].hex() for h in level)
        lines.append(f"{indent}[{hashes}]")
    return "\n".join(lines)


# Tests


@pytest.mark.parametrize(
    "data",
    [
        [b"a"],
        [b"a", b"b"],
        [b"a", b"b", b"c"],
        [b"a", b"b", b"c", b"d"],
        [b"a", b"b", b"c", b"d", b"e"],
        [b"a", b"b", b"c", b"d", b"e", b"f", b"g"],
        [b"a", b"b", b"c", b"d", b"e", b"f", b"g", b"h"],
    ],
)
def test_build_and_verify_proof(data: list[bytes]) -> None:
    tree = build_tree(data)
    assert tree is not None

    for i, block in enumerate(data):
        proof = generate_proof(tree, i)
        assert proof is not None, f"Proof generation failed for index {i}"
        assert verify_proof(tree.root, block, proof), (
            f"Verification failed for index {i}"
        )


def test_invalid_proof_wrong_data() -> None:
    tree = build_tree([b"a", b"b", b"c", b"d"])
    assert tree is not None

    proof = generate_proof(tree, 0)
    assert proof is not None
    assert not verify_proof(tree.root, b"wrong", proof)


def test_invalid_proof_wrong_root() -> None:
    tree = build_tree([b"a", b"b", b"c", b"d"])
    assert tree is not None

    proof = generate_proof(tree, 0)
    assert proof is not None
    assert not verify_proof(b"wrong_root_hash", b"a", proof)


def test_diff_identical_trees() -> None:
    data = [b"a", b"b", b"c", b"d"]
    assert diff_trees(build_tree(data), build_tree(data)) == []


def test_diff_single_change() -> None:
    assert diff_trees(
        build_tree([b"a", b"b", b"c", b"d"]),
        build_tree([b"a", b"X", b"c", b"d"]),
    ) == [1]


def test_diff_multiple_changes() -> None:
    assert sorted(
        diff_trees(
            build_tree([b"a", b"b", b"c", b"d"]),
            build_tree([b"X", b"b", b"Y", b"d"]),
        )
    ) == [0, 2]


def test_empty_tree() -> None:
    assert build_tree([]) is None


def test_single_leaf() -> None:
    tree = build_tree([b"only"])
    assert tree is not None

    proof = generate_proof(tree, 0)
    assert proof is not None
    assert verify_proof(tree.root, b"only", proof)


def test_proof_out_of_bounds() -> None:
    tree = build_tree([b"a", b"b", b"c"])
    assert tree is not None
    assert generate_proof(tree, -1) is None
    assert generate_proof(tree, 3) is None
    assert generate_proof(tree, 100) is None


# CLI

cli = click.Group()


@cli.command()
@click.option("--blocks", "-n", default=8, help="Number of data blocks")
def demo(blocks: int) -> None:
    """Build a Merkle tree and demonstrate proof verification."""
    data = [f"block_{i}".encode() for i in range(blocks)]

    click.echo(f"Building Merkle tree with {blocks} blocks...")
    tree = build_tree(data)

    if tree is None:
        click.echo("Empty tree")
        return

    click.echo(f"\nRoot hash: {tree.root.hex()[:16]}...")
    click.echo(f"Tree depth: {len(tree.levels)}")
    click.echo("\nTree structure (hashes truncated):")
    click.echo(tree_to_str(tree))

    test_idx = blocks // 2
    click.echo(
        f"\nGenerating proof for block {test_idx} ('{data[test_idx].decode()}')..."
    )
    proof = generate_proof(tree, test_idx)

    if proof:
        click.echo(f"Proof path length: {len(proof.path)}")
        click.echo(
            f"Verification: {'VALID' if verify_proof(tree.root, data[test_idx], proof) else 'INVALID'}"
        )

        click.echo("\nTesting with tampered data...")
        click.echo(
            f"Tampered verification: {'VALID' if verify_proof(tree.root, b'tampered', proof) else 'INVALID'}"
        )


@cli.command()
@click.option("--blocks", "-n", default=8, help="Number of data blocks")
@click.option("--changes", "-c", default=2, help="Number of blocks to change")
def diff(blocks: int, changes: int) -> None:
    """Demonstrate tree diffing to find changed blocks."""
    import random

    data1 = [f"block_{i}".encode() for i in range(blocks)]
    data2 = list(data1)

    changed_indices = random.sample(range(blocks), min(changes, blocks))
    for idx in changed_indices:
        data2[idx] = f"modified_{idx}".encode()

    click.echo(f"Original: {blocks} blocks")
    click.echo(f"Modified blocks: {sorted(changed_indices)}")

    tree1 = build_tree(data1)
    tree2 = build_tree(data2)

    found = diff_trees(tree1, tree2)
    click.echo(f"\nDiff found: {sorted(found)}")
    click.echo(f"Match: {sorted(found) == sorted(changed_indices)}")


@cli.command("test")
def run_tests() -> None:
    """Run pytest on this module."""
    import subprocess
    import sys

    raise SystemExit(
        subprocess.run(
            [sys.executable, "-m", "pytest", __file__, "-v", "-p", "no:cacheprovider"],
            cwd=Path(__file__).parent,
        ).returncode
    )


if __name__ == "__main__":
    cli()
