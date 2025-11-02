#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.loguru -p python313Packages.pytest -p python313Packages.pyvis -p python313Packages.more-itertools -p python313Packages.click
"""
Floyd-Warshall algorithm for finding all shortest paths in a graph.
It is a dynamic programming algorithm that runs in O(n^3) time.
"""

from __future__ import annotations

import sys
import webbrowser
from collections.abc import Mapping, Sequence
from pathlib import Path

import click
import pytest

sys.path.append(str(Path(__file__).parent))
from graph import Graph, H
from viz import visualise_graph


def floyd_warshall(
    graph: Graph[H],
) -> tuple[Sequence[Sequence[float]], Mapping[H, int]]:
    nodes = list(graph.nodes.keys())
    n = len(nodes)
    node_to_index = {node: i for i, node in enumerate(nodes)}
    dist = [[0 if i == j else float("inf") for j in range(n)] for i in range(n)]

    for node, neighbor, weight in graph.edges:
        dist[node_to_index[node.key]][node_to_index[neighbor.key]] = weight

    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    return dist, node_to_index


def is_shortest_path_edge(
    graph: Graph[H],
    distances: Sequence[Sequence[float]],
    i: int,
    k: int,
    j: int,
) -> bool:
    nodes = list(graph.nodes.keys())
    return (
        k != j
        and distances[i][k] != float("inf")
        and graph[nodes[k]].adj.get_weight(graph[nodes[j]]) != float("inf")
        and abs(
            distances[i][k]
            + graph[nodes[k]].adj.get_weight(graph[nodes[j]])
            - distances[i][j]
        )
        < 1e-9
    )


def create_shortest_path_graph(
    graph: Graph[H],
    source: H,
    distances: Sequence[Sequence[float]],
    node_to_index: Mapping[H, int],
) -> Graph[H]:
    nodes = list(graph.nodes.keys())
    n = len(nodes)
    source_idx = node_to_index[source]
    result = Graph(nodes=nodes)

    for i in range(n):
        if distances[source_idx][i] != float("inf"):
            current = i
            path = [current]
            while current != source_idx:
                found = False
                for k in range(n):
                    if is_shortest_path_edge(graph, distances, source_idx, k, current):
                        current = k
                        path.append(current)
                        found = True
                        break
                if not found:
                    break

            for j in range(len(path) - 1):
                from_node = nodes[path[j + 1]]
                to_node = nodes[path[j]]
                weight = graph[from_node].adj.get_weight(graph[to_node])
                if weight != float("inf"):
                    result.add_edge(from_node, to_node, weight)

    return result


@pytest.mark.parametrize(
    "graph, expected_distances",
    [
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
            Graph(edges=[(0, 1, 1), (0, 2, 4), (1, 3, 2), (2, 3, 1)]),
            [
                [0, 1, 4, 3],
                [float("inf"), 0, float("inf"), 2],
                [float("inf"), float("inf"), 0, 1],
                [float("inf"), float("inf"), float("inf"), 0],
            ],
        ),
        (
            Graph(edges=[(0, 1, 2), (1, 2, 1), (2, 0, 3)]),
            [
                [0, 2, 3],
                [4, 0, 1],
                [3, 5, 0],
            ],
        ),
    ],
)
def test_floyd_warshall(graph: Graph[H], expected_distances: Sequence[Sequence[float]]):
    distances, _ = floyd_warshall(graph)
    assert distances == expected_distances


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
    default="floyd_warshall_shortest_paths.html",
    help="Output HTML filename",
)
@click.option(
    "--open",
    is_flag=True,
    default=False,
    help="Open the generated HTML file in the default browser",
)
def example(vertices: int, probability: float, seed: int, output: str, open: bool):
    random_graph = Graph.random(
        n=vertices, p=probability, seed=seed, index_to_name=lambda i: i
    )
    distances, node_to_index = floyd_warshall(random_graph)

    graphs = [random_graph]
    titles = ["Original Graph"]

    nodes = list(random_graph.nodes.keys())
    for source in nodes:
        shortest_path_graph = create_shortest_path_graph(
            random_graph, source, distances, node_to_index
        )
        graphs.append(shortest_path_graph)
        titles.append(f"Shortest paths from {source}")

    visualise_graph(graphs=graphs, output_filename=output, graph_titles=titles)

    click.echo(f"Floyd-Warshall visualization saved to: {output}")
    click.echo(
        f"Generated {len(graphs)} graphs: original + {len(nodes)} shortest path graphs"
    )

    if open:
        try:
            webbrowser.open(output)
            click.echo(f"Opening {output} in default browser...")
        except FileNotFoundError:
            click.echo(
                "Error: 'open' command not found. Please open the file manually."
            )


@cli.command("test")
def run_tests():
    pytest.main([__file__])


if __name__ == "__main__":
    cli()
