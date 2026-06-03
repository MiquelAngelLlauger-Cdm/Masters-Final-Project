"""
mapper.pipeline
===============

High-level orchestration. Wires the modules together so a full run is one call.

The lens-dependency ordering is the only subtlety:

* For **feature** and **density** lenses the lens does not depend on the graph,
  so the order is: distances -> lens -> cover -> graph.

* For the **centrality** lens the lens is computed *from* a proximity graph, so
  the order is: distances -> proximity graph -> lens -> cover -> Mapper graph.

``run_pipeline`` handles both transparently.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import networkx as nx

from . import data as data_mod
from . import distance as dist_mod
from . import lenses as lens_mod
from . import cover as cover_mod
from . import graph as graph_mod
from . import layout as layout_mod
from . import viz as viz_mod


@dataclass
class MapperResult:
    dataset: object        # Dataset
    D: np.ndarray          # distance matrix
    lens: object           # LensResult
    cover: object          # Cover
    graph: nx.Graph        # final Mapper graph (with layout positions)
    params: object         # MapperParams used


def run_pipeline(params, verbose: bool = True) -> MapperResult:
    """Run the entire pipeline and return all intermediate artefacts."""
    def log(*a):
        if verbose:
            print(*a)

    # 1. data
    ds = data_mod.load_dataset(params)
    log(data_mod.describe_dataset(ds, params), "\n")

    # 2. distances
    D = dist_mod.compute_distances(ds.X, params)
    log(dist_mod.describe_distances(D, params), "\n")

    # 3. lens (centrality needs a proximity graph first)
    G_prox = None
    if params.LENS_KIND.lower() == "centrality":
        G_prox = graph_mod.build_epsilon_graph(
            D, ds.df, cover=None, edge_rule=None, epsilon=params.EPSILON
        )
    lens = lens_mod.build_lens(params, ds.df, D, G_proximity=G_prox)
    log(lens.describe(), "\n")

    # 4. cover (tunable binning over the lens range)
    cover = cover_mod.build_cover(lens.values, params)
    log(cover_mod.describe_cover(cover, lens.values), "\n")

    # 5. Mapper graph with the configured edge rule
    G = graph_mod.build_epsilon_graph(
        D, ds.df, cover=cover, lens=lens,
        edge_rule=params.EDGE_RULE,
        epsilon=params.EPSILON, max_bin_gap=params.MAX_BIN_GAP,
    )
    log(graph_mod.describe_graph(G), "\n")

    # 6. layout
    layout_mod.compute_layout(G, ds.X, params)

    return MapperResult(dataset=ds, D=D, lens=lens, cover=cover,
                        graph=G, params=params)


def visualise(result: MapperResult, render: bool = True):
    """Render the Bokeh figure for a finished pipeline result."""
    return viz_mod.visualise(result.graph, result.dataset.df,
                             result.lens, result.params, render=render)
