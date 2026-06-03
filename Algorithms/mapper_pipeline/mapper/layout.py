"""
mapper.layout
=============

Node-position computation. Mirrors Section 5 of the notebook.
Supports spring / spectral / pca layouts. Positions are written back onto the
graph nodes as ``x`` / ``y`` attributes.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from sklearn.decomposition import PCA


def compute_layout(G: nx.Graph, X: np.ndarray, params) -> dict:
    """
    Compute and attach 2-D positions.

    Returns the position dict ``{node: (x, y)}`` and also stores ``x``/``y`` on
    each node for convenience.
    """
    layout = params.LAYOUT

    if layout == "spring":
        Gw = G.copy()
        for _, _, data in Gw.edges(data=True):
            data["spring_weight"] = 1.0 / (data["weight"] + 1e-6)
        pos = nx.spring_layout(
            Gw, weight="spring_weight",
            k=params.SPRING_K, iterations=params.SPRING_ITERS, seed=params.SEED,
        )

    elif layout == "spectral":
        pos = nx.spectral_layout(G)

    elif layout == "pca":
        pca = PCA(n_components=2, random_state=params.SEED)
        coords = pca.fit_transform(X)
        pos = {i: (float(coords[i, 0]), float(coords[i, 1]))
               for i in range(X.shape[0])}

    else:
        raise ValueError(f"Unknown LAYOUT '{layout}'.")

    for node, (x, y) in pos.items():
        G.nodes[node]["x"] = float(x)
        G.nodes[node]["y"] = float(y)

    return pos
