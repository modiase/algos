#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.loguru -p python313Packages.pytest -p python313Packages.click
"""
Rabin-based content-defined chunking (CDC) demo.

## Problem

Fixed-size chunking splits data at regular intervals. When data is inserted,
all subsequent chunk boundaries shift, causing nearly every chunk after the
insertion point to have a different fingerprint - even though most of the
underlying data is unchanged.

## Solution: Content-Defined Chunking

CDC uses a rolling hash (Rabin fingerprint) to find chunk boundaries based on
the data itself rather than fixed offsets. A boundary is placed wherever the
hash meets a condition (e.g., lowest 10 bits are zero).

Since boundaries are determined by local content, an insertion only affects:
1. The chunk containing the insertion
2. Possibly the next chunk (if the insertion changes where the boundary falls)

All other chunks remain identical, with matching fingerprints.

## Splitting Condition

Using mask 0x3FF tests if the lowest 10 bits are zero. Assuming uniform hash
distribution, P(split) = (1/2)^10 = 1/1024, yielding ~1KB average chunks.

## Results (from demo with 100KB data, 5 insertions of 50 bytes each)

    Fixed-size: 97/101 chunks changed (96.0% of data, 98KB)
    Rabin CDC:   6/80  chunks changed ( 8.0% of data,  8KB)

    Reduction: 91.7% less data to transfer/store

## Extension: Merkle Trees for O(log n) Diff

This demo compares fingerprints linearly - O(n) comparisons. For large datasets,
a Merkle tree over chunk fingerprints enables O(k log n) diff where k = changed
chunks:

    1. Leaf nodes = chunk fingerprints
    2. Internal nodes = hash of children's hashes
    3. Compare roots first - if equal, nothing changed
    4. Recursively descend only into differing subtrees

For 100 chunks with 6 changes: flat requires 100 comparisons, Merkle requires
~42 (6 changes × 7 tree depth).

## Real-World Applications

- rsync: rolling checksums for efficient file sync
- Git: Merkle trees for commits/trees/blobs
- IPFS: Merkle DAGs for content-addressed storage
- ZFS/Btrfs: Merkle trees for data integrity
- Backup systems (restic, borg): CDC for deduplication
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence

import click
import pytest
from loguru import logger

BASE = 256
MODULUS = (1 << 31) - 1
DEFAULT_MASK = 0x3FF

Boundaries = Sequence[tuple[int, int]]


def generate_content(seed: int, size: int) -> bytes:
    rng = random.Random(seed)
    return bytes(rng.randint(0, 255) for _ in range(size))


def fixed_boundaries(length: int, chunk_size: int) -> Boundaries:
    return tuple((i, min(i + chunk_size, length)) for i in range(0, length, chunk_size))


def rabin_hash(data: bytes) -> int:
    h = 0
    for byte in data:
        h = (h * BASE + byte) % MODULUS
    return h


def rb_boundaries(
    data: bytes,
    min_size: int = 512,
    max_size: int = 2048,
    mask: int = DEFAULT_MASK,
    window_size: int = 48,
) -> Boundaries:
    n = len(data)
    if n == 0:
        return tuple()

    bounds: list[tuple[int, int]] = []
    start = 0

    while start < n:
        remaining = n - start

        if remaining <= min_size:
            bounds.append((start, n))
            break

        end = start + min_size
        h = (
            rabin_hash(data[end - window_size : end])
            if end >= window_size
            else rabin_hash(data[start:end])
        )

        while end < n and end - start < max_size:
            if h & mask == 0:
                break
            old_byte = data[end - window_size] if end >= window_size else 0
            new_byte = data[end]
            highest_power = pow(BASE, window_size - 1, MODULUS)
            h = ((h - old_byte * highest_power) * BASE + new_byte) % MODULUS
            end += 1

        bounds.append((start, end))
        start = end

    return tuple(bounds)


def fingerprint_boundaries(data: bytes, bounds: Boundaries) -> Sequence[str]:
    return tuple(
        hashlib.sha256(data[start:end]).hexdigest()[:16] for start, end in bounds
    )


def chunk_sizes(bounds: Boundaries) -> Sequence[int]:
    return tuple(end - start for start, end in bounds)


def insert_random_data(
    data: bytes, seed: int, num_insertions: int, insert_size: int
) -> bytes:
    rng = random.Random(seed)
    result = bytearray(data)
    positions = sorted(rng.sample(range(len(data)), num_insertions), reverse=True)

    for pos in positions:
        insert_data = bytes(rng.randint(0, 255) for _ in range(insert_size))
        result[pos:pos] = insert_data

    return bytes(result)


def compare_fingerprints(
    original_fps: Sequence[str],
    modified_fps: Sequence[str],
    modified_sizes: Sequence[int],
) -> tuple[set[int], int]:
    original_set = set(original_fps)
    changed_indices: set[int] = set()
    changed_bytes = 0

    for i, fp in enumerate(modified_fps):
        if fp not in original_set:
            changed_indices.add(i)
            changed_bytes += modified_sizes[i]

    return changed_indices, changed_bytes


def render_diagram(changed_indices: set[int], total: int, width: int = 60) -> str:
    if total == 0:
        return "[]"

    chars: list[str] = []
    for i in range(width):
        chunk_idx = i * total // width
        chars.append("#" if chunk_idx in changed_indices else ".")

    return f"[{''.join(chars)}]"


cli = click.Group()


@cli.command()
@click.option("--seed", default=42, help="Random seed for data generation")
@click.option("--size", default=102400, help="Size of data in bytes (default: 100KB)")
@click.option("--chunk-size", default=1024, help="Fixed chunk size in bytes")
@click.option("--num-insertions", default=5, help="Number of insertions")
@click.option("--insert-size", default=50, help="Size of each insertion in bytes")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def demo(
    seed: int,
    size: int,
    chunk_size: int,
    num_insertions: int,
    insert_size: int,
    verbose: bool,
) -> None:
    """Run the RB chunking demonstration."""
    logger.remove()
    if verbose:
        logger.add(lambda msg: click.echo(msg, err=True), level="DEBUG")

    click.echo("=== RB Chunking Demo ===\n")

    original = generate_content(seed, size)
    click.echo(f"Original data: {len(original)} bytes\n")

    fixed_orig_bounds = fixed_boundaries(len(original), chunk_size)
    fixed_orig_fps = fingerprint_boundaries(original, fixed_orig_bounds)
    click.echo(f"Fixed-size chunking ({chunk_size} bytes):")
    click.echo(f"  Chunks: {len(fixed_orig_bounds)}")

    rb_orig_bounds = rb_boundaries(original)
    rb_orig_fps = fingerprint_boundaries(original, rb_orig_bounds)
    rb_orig_sizes = chunk_sizes(rb_orig_bounds)
    avg_rb_size = sum(rb_orig_sizes) // len(rb_orig_sizes) if rb_orig_sizes else 0
    click.echo("\nRB content-defined chunking:")
    click.echo(f"  Chunks: {len(rb_orig_bounds)}")
    click.echo(f"  Avg size: {avg_rb_size} bytes")

    total_inserted = num_insertions * insert_size
    click.echo(
        f"\nAfter inserting {num_insertions} chunks ({total_inserted} bytes total):"
    )

    modified = insert_random_data(original, seed + 1, num_insertions, insert_size)
    click.echo(f"Modified data: {len(modified)} bytes\n")

    fixed_mod_bounds = fixed_boundaries(len(modified), chunk_size)
    fixed_mod_fps = fingerprint_boundaries(modified, fixed_mod_bounds)
    fixed_mod_sizes = chunk_sizes(fixed_mod_bounds)
    fixed_changed, fixed_bytes = compare_fingerprints(
        fixed_orig_fps, fixed_mod_fps, fixed_mod_sizes
    )
    fixed_pct = 100 * fixed_bytes / len(modified)

    click.echo("Fixed-size chunking:")
    click.echo(f"  Chunks: {len(fixed_mod_bounds)}")
    click.echo(
        f"  Changed: {len(fixed_changed)} chunks ({fixed_bytes} bytes, {fixed_pct:.1f}% of data)"
    )
    click.echo(f"\n  {render_diagram(fixed_changed, len(fixed_mod_bounds))}")

    rb_mod_bounds = rb_boundaries(modified)
    rb_mod_fps = fingerprint_boundaries(modified, rb_mod_bounds)
    rb_mod_sizes = chunk_sizes(rb_mod_bounds)
    rb_changed, rb_bytes = compare_fingerprints(rb_orig_fps, rb_mod_fps, rb_mod_sizes)
    rb_pct = 100 * rb_bytes / len(modified)

    click.echo("\nRB content-defined chunking:")
    click.echo(f"  Chunks: {len(rb_mod_bounds)}")
    click.echo(
        f"  Changed: {len(rb_changed)} chunks ({rb_bytes} bytes, {rb_pct:.1f}% of data)"
    )
    click.echo(f"\n  {render_diagram(rb_changed, len(rb_mod_bounds))}")

    click.echo("\nLegend: . = unchanged, # = changed")

    if fixed_bytes > 0:
        reduction = 100 * (1 - rb_bytes / fixed_bytes)
        click.echo(f"\nConclusion: RB CDC reduced changed data by {reduction:.1f}%")


@pytest.mark.parametrize(
    ("length", "chunk_size", "expected_count"),
    [
        (100, 10, 10),
        (105, 10, 11),
        (0, 10, 0),
    ],
)
def test_fixed_boundaries(length: int, chunk_size: int, expected_count: int) -> None:
    bounds = fixed_boundaries(length, chunk_size)
    assert len(bounds) == expected_count
    if bounds:
        assert bounds[0][0] == 0
        assert bounds[-1][1] == length


def test_rb_boundaries_covers_all_data() -> None:
    data = generate_content(42, 10000)
    bounds = rb_boundaries(data)
    assert bounds[0][0] == 0
    assert bounds[-1][1] == len(data)
    for i in range(len(bounds) - 1):
        assert bounds[i][1] == bounds[i + 1][0]


def test_rb_boundaries_respects_size_limits() -> None:
    data = generate_content(42, 10000)
    bounds = rb_boundaries(data, min_size=500, max_size=2000)
    sizes = chunk_sizes(bounds)
    for size in sizes[:-1]:
        assert 500 <= size <= 2000


def test_fingerprint_deterministic() -> None:
    data = b"test data"
    bounds = ((0, len(data)),)
    fp1 = fingerprint_boundaries(data, bounds)
    fp2 = fingerprint_boundaries(data, bounds)
    assert fp1 == fp2


def test_insert_random_data() -> None:
    original = b"0123456789"
    modified = insert_random_data(original, 42, 2, 3)
    assert len(modified) == len(original) + 6


@cli.command("test")
def run_tests() -> None:
    """Run tests."""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    cli()
