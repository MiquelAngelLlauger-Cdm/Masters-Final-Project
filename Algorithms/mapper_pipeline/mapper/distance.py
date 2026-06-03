"""
mapper.distance
===============

Pairwise distance computation. Mirrors Section 3 of the notebook.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import pairwise_distances


def compute_distances(X: np.ndarray, params) -> np.ndarray:
    """
    Compute the (n, n) pairwise distance matrix using the configured metric.

    Honours ``params.METRIC`` and (for minkowski) ``params.MINKOWSKI_P``.
    """
    D = pairwise_distances(X, metric=params.METRIC, **params.dist_kwargs)
    return D


def describe_distances(D: np.ndarray, params) -> str:
    """Percentile summary + edge-count preview at the chosen epsilon."""
    n = D.shape[0]
    off = D[D > 0]
    lines = [
        f"Distance matrix : {D.shape}",
        f"Distance range  : [{D.min():.3f}, {D.max():.3f}]",
        "Percentiles:",
    ]
    for p in (10, 25, 50, 75, 90):
        lines.append(f"  {p:3d}th : {np.percentile(off, p):.3f}")
    n_edges_raw = int(((D <= params.EPSILON) & (D > 0)).sum() // 2)
    density = n_edges_raw / (n * (n - 1) / 2) * 100
    lines += [
        f"With ε = {params.EPSILON}:",
        f"  Edges before pruning : {n_edges_raw}",
        f"  Edge density         : {density:.1f}%",
    ]
    return "\n".join(lines)
