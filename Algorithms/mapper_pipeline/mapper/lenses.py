"""
mapper.lenses
=============

Filter / projection functions ("lenses") for the Mapper pipeline.

A lens maps each patient to a real value f(i) -> R. The cover is then built
over the *range* of the lens (see ``cover.py``), and edges are pruned so that
two proximate patients are connected only if they share a cover set.

Three families of lens are provided:

1. **Feature lens** (`feature_lens`) — the data-driven projection used in the
   original notebook (e.g. ``attendance_density_w8``). Just reads a column.

2. **Density lens** (`density_lens`) — projects each node onto a local-density
   estimate in feature space (kNN distance, ε-ball count, or a Gaussian KDE
   surrogate). This is a geometry-driven lens: dense cores vs. sparse outliers.

All lenses return a 1-D ``numpy`` array of length ``n_patients`` plus a short
human-readable name (used in plot titles / axis labels).

The lens value is what gets *binned* by the cover; the binning itself stays
fully tunable via ``cover.py`` and the parameters object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import networkx as nx


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# 1. Feature lens (data-driven) — original notebook behaviour
# --------------------------------------------------------------------------- #
def feature_lens(df, column: str) -> LensResult:
    """
    Project patients onto a raw data column.

    This reproduces the original notebook's filter function, e.g.
    ``feature_lens(df, "attendance_density_w8")``.

    Parameters
    ----------
    df : pandas.DataFrame
        Patient table (one row per patient, aligned with the distance matrix).
    column : str
        Column name to use as the lens.
    """
    if column not in df.columns:
        raise KeyError(f"Feature lens column '{column}' not in dataframe.")
    values = df[column].to_numpy(dtype=float)
    return LensResult(values=values, name=column, kind="feature")



# --------------------------------------------------------------------------- #
# 3. Density lens (geometry-driven)
# --------------------------------------------------------------------------- #
def density_lens(
    D: np.ndarray,
    method: str = "knn",
    k: int = 10,
    epsilon: Optional[float] = None,
    bandwidth: Optional[float] = None,
) -> LensResult:
    """
    Project patients onto a local-density estimate in feature space.

    Works directly off the precomputed pairwise distance matrix ``D`` so it is
    consistent with whatever ``METRIC`` was used to build the graph.

    Parameters
    ----------
    D : numpy.ndarray
        Symmetric (n, n) pairwise distance matrix.
    method : str
        - ``"kde"``      : Gaussian-kernel density surrogate using ``bandwidth``.
                           High value = dense region.
    k : int
        Number of neighbours for ``knn`` / ``knn_dist``.
    epsilon : float, optional
        Radius for ``ball`` method (defaults to the median off-diagonal dist).
    bandwidth : float, optional
        Gaussian bandwidth for ``kde`` (defaults to the median off-diagonal
        distance, a robust Silverman-ish surrogate).

    Notes
    -----
    The density lens is computed from distances only; it does not require the
    raw feature matrix, which keeps it metric-agnostic.
    """
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


# --------------------------------------------------------------------------- #
# Dispatcher — single entry point used by the pipeline
# --------------------------------------------------------------------------- #
def build_lens(params, df, D, G_proximity: Optional[nx.Graph] = None) -> LensResult:
    """
    Construct the lens specified by ``params.LENS_KIND``.

    Parameters
    ----------
    params : MapperParams
        Configuration (see ``config.py``). Relevant fields:
        - LENS_KIND          : "feature" | "centrality" | "density"
        - FEATURE_LENS_COL   : column name (feature lens)
        - CENTRALITY_MEASURE : degree/betweenness/... (centrality lens)
        - DENSITY_METHOD     : knn/knn_dist/ball/kde   (density lens)
        - DENSITY_K          : k for knn methods
        - DENSITY_BANDWIDTH  : bandwidth for kde
    df : pandas.DataFrame
    D  : numpy.ndarray
        Pairwise distance matrix.
    G_proximity : networkx.Graph, optional
        Required for the centrality lens. Should be the proximity-only graph.
    """
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
