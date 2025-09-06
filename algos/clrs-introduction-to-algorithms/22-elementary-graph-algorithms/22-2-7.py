#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.pytest
"""
Wrestler designation via BFS 2-coloring (bipartite test), kept minimal.

- Input: vertices (optional) and undirected edges (rivalries).
- Output: (babyfaces, heels) sets if a valid split exists; otherwise None.
- Idea: BFS-color each connected component with colors {0,1}.
        If we ever see an edge whose endpoints share a color, return None.
Time: O(n + r)
"""

import sys
from collections import deque
from typing import Hashable, Iterable, Optional

import pytest


def bipartition_wrestlers(
    rivalries: Iterable[tuple[Hashable, Hashable]],
    vertices: Optional[Iterable[Hashable]] = None,
) -> Optional[tuple[set[Hashable], set[Hashable]]]:
    """
    BFS 2-coloring without any explicit "cycle" logic—just color neighbors
    oppositely and fail if an edge connects same-colored endpoints.

    Args:
        rivalries: iterable of undirected pairs (u, v)
        vertices: optional set/list of all vertices; if omitted, inferred from edges

    Returns:
        (babyfaces, heels) if bipartite else None
    """
    adj: dict[Hashable, set[Hashable]] = {}
    all_v: set[Hashable] = set(vertices) if vertices is not None else set()
    for u, v in rivalries:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
        if vertices is None:
            all_v.add(u)
            all_v.add(v)

    for v in list(all_v):
        adj.setdefault(v, set())

    if vertices is not None:
        for v in vertices:
            adj.setdefault(v, set())
        all_v = set(adj.keys())

    color: dict[Hashable, int] = {}

    for s in all_v:
        if s in color:
            continue
        color[s] = 0
        q: deque = deque([s])

        while q:
            u = q.popleft()
            for w in adj[u]:
                if w not in color:
                    color[w] = 1 - color[u]
                    q.append(w)
                elif color[w] == color[u]:
                    return None

    babyfaces = {v for v, c in color.items() if c == 0}
    heels = set(all_v) - babyfaces
    return babyfaces, heels


@pytest.mark.parametrize(
    "vertices, edges, expect_bipartite",
    [
        (["A", "B", "C", "D"], [("A", "B"), ("B", "C"), ("C", "D")], True),
        ([0, 1, 2, 3], [(0, 1), (1, 2), (2, 3), (3, 0)], True),
        ([0, 1, 2], [(0, 1), (1, 2), (2, 0)], False),
        ([0, 1, 2], [(0, 1)], True),
        ([1, 2], [(1, 1)], False),
    ],
)
def test_bipartition_parametrized(
    vertices: Iterable[Hashable],
    edges: Iterable[tuple[Hashable, Hashable]],
    expect_bipartite: bool,
):
    res = bipartition_wrestlers(edges, vertices=vertices)
    if not expect_bipartite:
        assert res is None
    else:
        assert res is not None
        baby, _ = res
        for u, v in edges:
            assert (u in baby) ^ (v in baby), f"Edge {(u, v)} does not cross the cut"


if __name__ == "__main__":
    pytest.main([__file__, *sys.argv[1:]])
