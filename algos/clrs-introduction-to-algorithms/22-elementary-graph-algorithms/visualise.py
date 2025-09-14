#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.pyvis -p python313Packages.loguru python313Packages.pytest python313Packages.more-itertools
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Final

from loguru import logger
from pyvis.network import Network

sys.path.append(Path(__file__).parent)
from components import components
from graph import Graph

TMP_DIR: Final = Path("/tmp")


def visualize_graph(graph: Graph, output_filename: str) -> None:
    net = Network(
        height="600px",
        width="100%",
        bgcolor="#222222",
        font_color="white",
        directed=True,
    )

    net.set_options("""
    var options = {
      "physics": {
        "enabled": true,
        "stabilization": {"iterations": 100},
        "barnesHut": {
          "gravitationalConstant": -2000,
          "centralGravity": 0.1,
          "springLength": 95,
          "springConstant": 0.04,
          "damping": 0.09
        }
      }
    }
    """)

    for node in graph:
        net.add_node(str(node.key), label=str(node.key), color="#97C2FC", size=20)

    for node in graph:
        for neighbor in node.adj:
            net.add_edge(str(node.key), str(neighbor.key), color="#848484", width=2)

    os.chdir(
        tempfile.gettempdir()
    )  # save_graph creates some library files in the current directory
    net.save_graph(output_filename)
    logger.info(f"Graph visualization saved to: {output_filename}")


if __name__ == "__main__":
    graph = Graph.random(n=10, p=0.4, namer=Graph.ascii_namer, seed=1)
    sccs = components(graph)
    node_to_scc_idx = {node: idx for idx, scc in enumerate(sccs) for node in scc}
    scc_forest = Graph()
    for node in graph:
        scc_forest.add_node(node.key)
        for neighbor in node.adj:
            if node_to_scc_idx[neighbor] == node_to_scc_idx[node]:
                scc_forest.add_edge(node.key, neighbor.key)

    output_file = TMP_DIR / f"graph_visualization_{abs(hash(graph))}.html"
    visualize_graph(graph, str(output_file))
    logger.success(f"Graph visualization available at: {output_file}")
    subprocess.run(f"open {output_file}", shell=True)

    output_file = TMP_DIR / f"graph_visualization_components_{abs(hash(graph))}.html"
    visualize_graph(scc_forest, str(output_file))
    logger.success(f"Graph visualization available at: {output_file}")
    subprocess.run(f"open {output_file}", shell=True)
