#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.pyvis -p python313Packages.loguru python313Packages.pytest python313Packages.more-itertools
from __future__ import annotations

import os
import sys
import tempfile
from collections.abc import Collection
from pathlib import Path
from typing import Final

from loguru import logger
from pyvis.network import Network

sys.path.append(str(Path(__file__).parent))
from graph import Graph

TMP_DIR: Final = Path("/tmp")


def _post_process_html(file_path: Path) -> None:
    with open(file_path, "r") as f:
        content = f.read()

    content = content.replace(
        "</head>",
        """
    <style>
    body {
        background-color: #222222 !important;
        margin: 20px auto !important;
        padding: 20px !important;
        max-width: 95% !important;
    }
    #mynetworkid {
        margin: 20px auto !important;
        border: 1px solid #444444 !important;
        border-radius: 8px !important;
    }
    </style>
    """
        + "</head>",
    )

    with open(file_path, "w") as f:
        f.write(content)


def visualise_graph(
    graphs: Collection[Graph],
    output_filename: str | Path,
    graph_titles: Collection[str] | None = None,
) -> None:
    node_colors = [
        "#E3F2FD",
        "#F3E5F5",
        "#E8F5E8",
        "#FFF3E0",
        "#FCE4EC",
        "#E0F2F1",
        "#F1F8E9",
        "#FFF8E1",
    ]
    edge_colors = [
        "#1976D2",
        "#7B1FA2",
        "#388E3C",
        "#F57C00",
        "#C2185B",
        "#00796B",
        "#689F38",
        "#FBC02D",
    ]

    net = Network(
        height="800px",
        width="100%",
        bgcolor="#222222",
        font_color="white",
        directed=True,
    )

    net.set_options("""
    var options = {
      "physics": {
        "enabled": true,
        "stabilization": {"iterations": 150},
        "barnesHut": {
          "gravitationalConstant": -1000,
          "centralGravity": 0.1,
          "springLength": 150,
          "springConstant": 0.02,
          "damping": 0.09
        }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 200
      },
      "layout": {
        "improvedLayout": true
      }
    }
    """)

    for graph_idx, graph in enumerate(graphs):
        net.add_node(
            f"gravity_{graph_idx}",
            label="",
            color="transparent",
            size=1,
            x=(graph_idx % 3) * 400 - 400,
            y=(graph_idx // 3) * 400 - 400,
            physics=False,
            hidden=True,
        )

        for node in graph:
            net.add_node(
                f"g{graph_idx}_{node.key}",
                label=f"{node.key}"
                + (
                    f" ({list(graph_titles)[graph_idx]})"
                    if graph_titles and graph_idx < len(graph_titles)
                    else ""
                ),
                color=node_colors[graph_idx % 8],
                size=20,
                title=f"Graph {graph_idx + 1}: {node.key}",
                group=graph_idx,
            )

            net.add_edge(
                f"gravity_{graph_idx}",
                f"g{graph_idx}_{node.key}",
                color="transparent",
                width=0,
                hidden=True,
                physics=True,
                length=150,
                springConstant=0.01,
            )

        for node in graph:
            for neighbor, weight in node.adj:
                net.add_edge(
                    f"g{graph_idx}_{node.key}",
                    f"g{graph_idx}_{neighbor.key}",
                    color=edge_colors[graph_idx % 8],
                    width=2,
                    label=f"{weight:.1f}",
                    title=f"Graph {graph_idx + 1}: {node.key} → {neighbor.key} (weight: {weight})",
                )

    os.chdir(tempfile.gettempdir())
    output_path = (
        output_filename
        if isinstance(output_filename, Path)
        else Path(output_filename).resolve()
    )
    net.save_graph(str(output_path))

    _post_process_html(output_path)
    logger.info(f"Graph visualisation saved to: {output_path}")


def visualise_single_graph(graph: Graph, output_filename: str | Path) -> None:
    """Convenience function for visualising a single graph (backward compatibility)."""
    visualise_graph([graph], output_filename)
