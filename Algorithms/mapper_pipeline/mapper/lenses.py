"""
mapper.lenses
=============

Filter / projection functions ("lenses") for the Metric Graph pipeline.

Two families of lens are provided:

1. **Feature lens** (`feature_lens`) 
2. **Density lens** (`density_lens`) — Gaussian KDE
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import networkx as nx


@dataclass
class LensResult:
    """Output of a lens: the per-patient values plus metadata for plotting."""
    values: np.ndarray            # shape (n,), float; NaN allowed for missing
    name: str                     # short label, e.g. "attendance_density_w8"
    kind: str                     # "feature" | "centrality" | "density"
    detail: str = ""              # extra info, e.g. "betweenness" or "knn(k=10)"

    @property
    def label(self) -> str:
        return f"{self.name}" if not self.detail else f"{self.name} ({self.detail})"

    def describe(self) -> str:
        v = self.values[~np.isnan(self.values)]
        if v.size == 0:
            return f"[{self.kind}] {self.label}: all-NaN"
        return (f"[{self.kind}] {self.label}: "
                f"min={v.min():.3f} median={np.median(v):.3f} "
                f"max={v.max():.3f} (n_valid={v.size})")



# 1. Feature lens (data-driven) 
# --------------------------------------------------------------------------- #
def feature_lens(df, column: str) -> LensResult:
 
    if column not in df.columns:
        raise KeyError(f"Feature lens column '{column}' not in dataframe.")
    values = df[column].to_numpy(dtype=float)
    return LensResult(values=values, name=column, kind="feature")



# 3. Density lens (geometry-driven)
# --------------------------------------------------------------------------- #
def density_lens(
    D: np.ndarray,
    method: str = "kde",
    k: int = 10,
    epsilon: Optional[float] = None,
    bandwidth: Optional[float] = None,
) -> LensResult:
    
    n = D.shape[0]
    method = method.lower()

    # robust default scale
    off = D[~np.eye(n, dtype=bool)]
    median_d = float(np.median(off)) if off.size else 1.0

    if method == "kde":
        h = bandwidth if bandwidth is not None else median_d
        # Gaussian kernel density surrogate (unnormalised but monotone)
        values = np.exp(-(D ** 2) / (2.0 * h ** 2)).sum(axis=1) - 1.0
        detail = f"gaussian kde h={h:.3f}"

    else:
        raise ValueError(
            f"Unknown density method '{method}'. "
            f"Choose from: knn, knn_dist, ball, kde."
        )

    return LensResult(values=values, name="density",
                      kind="density", detail=detail)


# Dispatcher — single entry point used by the pipeline
# --------------------------------------------------------------------------- #
def build_lens(params, df, D, G_proximity: Optional[nx.Graph] = None) -> LensResult:
    kind = params.LENS_KIND.lower()

    if kind == "feature":
        return feature_lens(df, params.FEATURE_LENS_COL)
    
    if kind == "density":
        return density_lens(
            D,
            method=params.DENSITY_METHOD,
            k=params.DENSITY_K,
            epsilon=params.EPSILON if params.DENSITY_METHOD == "ball" else None,
            bandwidth=params.DENSITY_BANDWIDTH,
        )

    raise ValueError(
        f"Unknown LENS_KIND '{params.LENS_KIND}'. "
        f"Choose from: feature, centrality, density."
    )
