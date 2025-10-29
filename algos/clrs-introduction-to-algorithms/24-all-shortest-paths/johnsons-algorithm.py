#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.loguru -p python313Packages.pytest -p python313Packages.pyvis -p python313Packages.more-itertools -p python313Packages.click
"""
Johnson's algorithm for all-pairs shortest paths on sparse graphs.
It combines Bellman-Ford with repeated Dijkstra runs.
"""

from __future__ import annotations

import sys
import webbrowser
from collections.abc import Hashable, Mapping, Sequence
from heapq import heappop, heappush
from pathlib import Path
from types import MappingProxyType

import click
import pytest

sys.path.append(str(Path(__file__).parent))
from graph import Graph, H
from viz import visualise_graph

_Adjacency = Mapping[H, Sequence[tuple[H, float]]]


def _compute_potentials(graph: Graph[H], nodes: Sequence[H]) -> Mapping[H, float]:
    super_source: Hashable = object()
    all_nodes: list[Hashable] = [*nodes, super_source]
    node_to_distance: dict[Hashable, float] = {node: float("inf") for node in all_nodes}
    node_to_distance[super_source] = 0.0

    edges: list[tuple[Hashable, Hashable, float]] = [
        (node.key, neighbor.key, weight) for node, neighbor, weight in graph.edges
    ]
    for node in nodes:
        edges.append((super_source, node, 0.0))

    for _ in range(len(all_nodes) - 1):
        updated = False
        for u, v, weight in edges:
            if node_to_distance[u] + weight < node_to_distance[v]:
                node_to_distance[v] = node_to_distance[u] + weight
                updated = True
        if not updated:
            break

    for u, v, weight in edges:
        if node_to_distance[u] + weight < node_to_distance[v]:
            raise ValueError("Graph contains a negative weight cycle")

    return MappingProxyType({node: node_to_distance[node] for node in nodes})


def _reweight_edges(
    graph: Graph[H],
    nodes: Sequence[H],
    potentials: Mapping[H, float],
) -> Mapping[H, Sequence[tuple[H, float]]]:
    node_to_edges: dict[H, list[tuple[H, float]]] = {node: [] for node in nodes}
    for node, neighbor, weight in graph.edges:
        new_weight = weight + potentials[node.key] - potentials[neighbor.key]
        node_to_edges[node.key].append((neighbor.key, new_weight))
    return MappingProxyType(
        {node: tuple(edges) for node, edges in node_to_edges.items()}
    )


def _dijkstra(
    adjacency: _Adjacency,
    nodes: Sequence[H],
    source: H,
) -> tuple[Mapping[H, float], Mapping[H, H | None]]:
    node_to_distance: dict[H, float] = {node: float("inf") for node in nodes}
    node_to_predecessor: dict[H, H | None] = {node: None for node in nodes}
    node_to_distance[source] = 0.0

    priority_queue: list[tuple[float, H]] = [(0.0, source)]
    processed_nodes: set[H] = set()

    while priority_queue:
        dist_u, u = heappop(priority_queue)
        if u in processed_nodes:
            continue
        processed_nodes.add(u)

        for v, weight in adjacency.get(u, ()):
            candidate = dist_u + weight
            if candidate < node_to_distance[v]:
                node_to_distance[v] = candidate
                node_to_predecessor[v] = u
                heappush(priority_queue, (candidate, v))

    return MappingProxyType(node_to_distance), MappingProxyType(node_to_predecessor)


def johnsons_algorithm(
    graph: Graph[H],
) -> tuple[
    Sequence[Sequence[float]],
    Mapping[H, int],
    Mapping[H, Mapping[H, float]],
    Mapping[H, Mapping[H, H | None]],
]:
    nodes = tuple(sorted(graph.nodes.keys(), key=str))
    node_to_index_dict = {node: index for index, node in enumerate(nodes)}

    node_to_potential = _compute_potentials(graph, nodes)
    node_to_edges = _reweight_edges(graph, nodes, node_to_potential)

    distance_rows: list[list[float]] = [
        [float("inf")] * len(nodes) for _ in range(len(nodes))
    ]
    source_to_node_to_distance: dict[H, Mapping[H, float]] = {}
    source_to_node_to_predecessor: dict[H, Mapping[H, H | None]] = {}

    for source in nodes:
        node_to_reweighted_distance, node_to_predecessor = _dijkstra(
            node_to_edges, nodes, source
        )
        node_to_adjusted_distance: dict[H, float] = {}

        for target in nodes:
            reweighted_distance = node_to_reweighted_distance[target]
            if reweighted_distance == float("inf"):
                adjusted_distance = float("inf")
            else:
                adjusted_distance = (
                    reweighted_distance
                    - node_to_potential[source]
                    + node_to_potential[target]
                )
            distance_rows[node_to_index_dict[source]][node_to_index_dict[target]] = (
                adjusted_distance
            )
            node_to_adjusted_distance[target] = adjusted_distance

        source_to_node_to_distance[source] = MappingProxyType(node_to_adjusted_distance)
        source_to_node_to_predecessor[source] = node_to_predecessor

    return (
        tuple(tuple(row) for row in distance_rows),
        MappingProxyType(node_to_index_dict),
        MappingProxyType(source_to_node_to_distance),
        MappingProxyType(source_to_node_to_predecessor),
    )


def create_shortest_path_graph(
    graph: Graph[H],
    source: H,
    node_to_distance: Mapping[H, float],
    node_to_predecessor: Mapping[H, H | None],
) -> Graph[H]:
    result = Graph(nodes=list(graph.nodes.keys()))

    for node, distance in node_to_distance.items():
        if node == source or distance == float("inf"):
            continue
        current = node
        seen: set[H] = set()
        while current != source and current not in seen:
            seen.add(current)
            predecessor = node_to_predecessor.get(current)
            if predecessor is None:
                break
            weight = graph[predecessor].adj.get_weight(graph[current])
            if weight == float("inf"):
                break
            result.add_edge(predecessor, current, weight)
            current = predecessor

    return result


@pytest.mark.parametrize(
    "graph, expected_distance_rows",
    [
        (
            Graph(
                edges=[
                    (0, 1, 3),
                    (0, 2, 8),
                    (0, 4, -4),
                    (1, 3, 1),
                    (1, 4, 7),
                    (2, 1, 4),
                    (3, 0, 2),
                    (3, 2, -5),
                    (4, 3, 6),
                ]
            ),
            [
                [0, 1, -3, 2, -4],
                [3, 0, -4, 1, -1],
                [7, 4, 0, 5, 3],
                [2, -1, -5, 0, -2],
                [8, 5, 1, 6, 0],
            ],
        ),
        (
            Graph(edges=[(0, 1, 5), (1, 2, 3), (2, 3, 2)]),
            [
                [0, 5, 8, 10],
                [float("inf"), 0, 3, 5],
                [float("inf"), float("inf"), 0, 2],
                [float("inf"), float("inf"), float("inf"), 0],
            ],
        ),
        (
            Graph(nodes=[0, 1, 2, 3], edges=[(0, 1, 2)]),
            [
                [0, 2, float("inf"), float("inf")],
                [float("inf"), 0, float("inf"), float("inf")],
                [float("inf"), float("inf"), 0, float("inf")],
                [float("inf"), float("inf"), float("inf"), 0],
            ],
        ),
    ],
)
def test_johnsons_algorithm(
    graph: Graph[H], expected_distance_rows: Sequence[Sequence[float]]
):
    distances, _, _, _ = johnsons_algorithm(graph)
    expected_rows = tuple(tuple(row) for row in expected_distance_rows)
    assert distances == expected_rows


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--vertices", "-n", default=5, help="Number of vertices in the random graph"
)
@click.option(
    "--probability", "-p", default=0.3, help="Edge probability for random graph"
)
@click.option("--seed", "-s", default=42, help="Random seed")
@click.option(
    "--output",
    "-o",
    default="johnson_shortest_paths.html",
    help="Output HTML filename",
)
@click.option(
    "--open",
    "open_browser",
    is_flag=True,
    default=False,
    help="Open the generated HTML file in the default browser",
)
def example(
    vertices: int, probability: float, seed: int, output: str, open_browser: bool
):
    random_graph = Graph.random(
        n=vertices, p=probability, seed=seed, index_to_name=lambda i: i
    )
    (
        _,
        _,
        source_to_node_to_distance,
        source_to_node_to_predecessor,
    ) = johnsons_algorithm(random_graph)

    graphs = [random_graph]
    titles = ["Original Graph"]

    for source, node_to_distance in source_to_node_to_distance.items():
        shortest_path_graph = create_shortest_path_graph(
            random_graph,
            source,
            node_to_distance,
            source_to_node_to_predecessor[source],
        )
        graphs.append(shortest_path_graph)
        titles.append(f"Shortest paths from {source}")

    visualise_graph(graphs=graphs, output_filename=output, graph_titles=titles)

    click.echo(f"Johnson's algorithm visualization saved to: {output}")
    click.echo(
        f"Generated {len(graphs)} graphs: original + {len(source_to_node_to_distance)} shortest path graphs"
    )

    if open_browser:
        try:
            webbrowser.open(output)
            click.echo(f"Opening {output} in default browser...")
        except FileNotFoundError:
            click.echo(
                "Error: 'open' command not found. Please open the file manually."
            )


@cli.command()
def test():
    pytest.main([__file__])


if __name__ == "__main__":
    cli()
