"""
mapper.cover
============

Open cover construction over the *range of the lens*, and assignment of each
patient to the cover sets it belongs to.

Two cover modes, both tunable from ``config.MapperParams``:

* ``"uniform"`` — the standard Mapper parameterisation: ``N_INTERVALS``
  overlapping intervals of equal width spanning the lens range, with fractional
  ``OVERLAP`` (a.k.a. "gain"). This generalises the notebook's hand-built
  26-interval cover.

* ``"edges"``   — explicit, possibly irregular bin edges (the notebook's
  ``BIN_EDGES`` / ``BIN_LABELS`` style). Useful for clinically meaningful,
  non-uniform bins (e.g. detox length-of-stay 0-3 / 3-7 / 7-14 / 14+ days).

The cover is what makes the binning tunable; nothing else needs to change to
re-bin the lens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass
class Cover:
    intervals: List[Tuple[float, float]]   # [lo, hi) per set
    labels: List[str]
    memberships: List[List[int]]           # per patient: indices of sets it's in
    bin_index: np.ndarray                  # per patient: primary (first) set, -1 if none
    mode: str

    @property
    def n_sets(self) -> int:
        return len(self.intervals)


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def build_uniform_cover(lens_values: np.ndarray,
                        n_intervals: int,
                        overlap: float,
                        rng: Tuple[float, float] | None = None
                        ) -> List[Tuple[float, float]]:
    """
    Standard Mapper cover: ``n_intervals`` equal-width overlapping intervals.

    ``overlap`` is the fractional overlap between adjacent intervals (gain).
    With overlap=0.5 adjacent intervals overlap by half their width, so most
    interior points land in exactly two sets.
    """
    finite = lens_values[np.isfinite(lens_values)]
    lo, hi = rng if rng is not None else (float(finite.min()), float(finite.max()))
    if hi <= lo:
        hi = lo + 1e-9
    span = hi - lo

    # Step between interval starts, and interval width, chosen to give the
    # requested overlap. width = step / (1 - overlap).
    step = span / n_intervals
    width = step / (1.0 - overlap) if overlap < 1.0 else step
    intervals = []
    for i in range(n_intervals):
        a = lo + i * step
        b = a + width
        # pad the final interval slightly so the max value is included
        if i == n_intervals - 1:
            b = max(b, hi + 1e-9)
        intervals.append((round(a, 6), round(b, 6)))
    return intervals


def build_edge_cover(bin_edges: List[float]) -> List[Tuple[float, float]]:
    """Explicit (possibly irregular) bins from consecutive edges."""
    return [(bin_edges[i], bin_edges[i + 1]) for i in range(len(bin_edges) - 1)]


# --------------------------------------------------------------------------- #
# Membership assignment
# --------------------------------------------------------------------------- #
def _memberships(values: np.ndarray,
                 intervals: List[Tuple[float, float]]) -> List[List[int]]:
    out = []
    for v in values:
        if not np.isfinite(v):
            out.append([])
            continue
        out.append([i for i, (lo, hi) in enumerate(intervals) if lo <= v < hi])
    return out


def build_cover(lens_values: np.ndarray, params) -> Cover:
    """
    Build the cover specified by ``params`` and assign memberships.

    Returns a fully populated ``Cover``.
    """
    if params.COVER_MODE == "uniform":
        intervals = build_uniform_cover(
            lens_values, params.N_INTERVALS, params.OVERLAP, params.COVER_RANGE
        )
        labels = [f"[{lo:.3g},{hi:.3g})" for lo, hi in intervals]
        memberships = _memberships(lens_values, intervals)
        bin_index = np.array([m[0] if m else -1 for m in memberships], dtype=int)

    elif params.COVER_MODE == "edges":
        if params.BIN_EDGES is None:
            raise ValueError("COVER_MODE='edges' requires BIN_EDGES.")
        intervals = build_edge_cover(params.BIN_EDGES)
        if params.BIN_LABELS is not None:
            labels = list(params.BIN_LABELS)
        else:
            labels = [f"[{lo:.3g},{hi:.3g})" for lo, hi in intervals]
        # explicit edges are non-overlapping -> use pandas.cut for the index
        idx = pd.cut(lens_values, bins=params.BIN_EDGES, labels=False,
                     right=params.BIN_RIGHT, include_lowest=True)
        bin_index = np.where(np.isnan(idx), -1, idx).astype(int)
        memberships = [[b] if b >= 0 else [] for b in bin_index]
    
    elif params.COVER_MODE == "piecewise":
        if not params.PIECEWISE_SEGMENTS:
            raise ValueError("COVER_MODE='piecewise' requires PIECEWISE_SEGMENTS.")
        intervals = build_piecewise_cover(params.PIECEWISE_SEGMENTS, params.OVERLAP)
        labels = [f"[{lo:.3g},{hi:.3g})" for lo, hi in intervals]
        memberships = _memberships(lens_values, intervals)
        bin_index = np.array([m[0] if m else -1 for m in memberships], dtype=int)

    elif params.COVER_MODE == "balanced":
        intervals = build_balanced_cover(lens_values, params.N_INTERVALS, params.OVERLAP)
        labels = [f"[{lo:.3g},{hi:.3g})" for lo, hi in intervals]
        memberships = _memberships(lens_values, intervals)
        bin_index = np.array([m[0] if m else -1 for m in memberships], dtype=int)

    else:
        raise ValueError(f"Unknown COVER_MODE '{params.COVER_MODE}'.")

    return Cover(intervals=intervals, labels=labels,
                 memberships=memberships, bin_index=bin_index,
                 mode=params.COVER_MODE)


def describe_cover(cover: Cover, lens_values: np.ndarray) -> str:
    """Membership distribution report (mirrors notebook Section 2)."""
    lines = [f"Cover mode: {cover.mode}  |  {cover.n_sets} sets", ""]
    for i, (lab, (lo, hi)) in enumerate(zip(cover.labels, cover.intervals)):
        n = sum(1 for m in cover.memberships if i in m)
        flag = "" if n > 0 else "  <- empty"
        lines.append(f"  Set {i:2d} {lab:<16s}: {n:3d} patients{flag}")
    # how many sets each patient lands in
    counts = {}
    for m in cover.memberships:
        counts[len(m)] = counts.get(len(m), 0) + 1
    lines.append("")
    lines.append("Patients by number of cover sets:")
    for k in sorted(counts):
        tag = "  <- disconnected!" if k == 0 else ""
        lines.append(f"  in {k} set(s): {counts[k]}{tag}")
    return "\n".join(lines)




def build_piecewise_cover(segments, overlap):
    """
    Piecewise-uniform cover: each segment gets its own uniform sub-cover at the
    requested resolution, with `overlap` applied within the segment.

    Segments should be contiguous: segment[i].hi == segment[i+1].lo.
    Within each segment, intervals overlap at `overlap`. Across segment
    boundaries there is NO automatic overlap — the last interval of segment k
    ends where segment k starts the next batch. That's usually fine because
    cover sets are defined by membership, not adjacency, but if you want the
    boundary to be smooth you can add a small overlap by extending the
    intervals manually (see notes below).
    """
    intervals = []
    for lo, hi, n in segments:
        if n < 1:
            continue
        span = hi - lo
        step = span / n
        width = step / (1.0 - overlap) if overlap < 1.0 else step
        for i in range(n):
            a = lo + i * step
            b = a + width
            if i == n - 1:
                b = max(b, hi + 1e-9)   # include the right edge of segment
            intervals.append((round(a, 6), round(b, 6)))
    return intervals


def build_balanced_cover(lens_values, n_intervals, overlap):
    """
    Quantile-based cover: interval edges are placed at evenly-spaced quantiles
    of the lens distribution, so each interval contains roughly the same number
    of patients. Overlap is then applied between adjacent intervals.
    """
    finite = lens_values[np.isfinite(lens_values)]
    if finite.size == 0:
        raise ValueError("Balanced cover: no finite lens values.")
    # n_intervals+1 quantile edges -> n_intervals base bins
    qs = np.linspace(0.0, 1.0, n_intervals + 1)
    edges = np.quantile(finite, qs)
    # de-duplicate (lens can have ties / plateaus) so we don't make zero-width bins
    edges = np.unique(edges)
    if len(edges) < 2:
        raise ValueError("Balanced cover: lens has too few distinct values.")

    # build overlapping intervals: each interval centred on a base bin, widened
    # to overlap into the neighbours by `overlap` fraction of the base width
    intervals = []
    for i in range(len(edges) - 1):
        a, b = float(edges[i]), float(edges[i + 1])
        width = b - a
        pad = (overlap / (2.0 * (1.0 - overlap))) * width if overlap < 1.0 else width
        lo = a - pad
        hi = b + pad
        if i == len(edges) - 2:
            hi = max(hi, float(edges[-1]) + 1e-9)
        intervals.append((round(lo, 6), round(hi, 6)))
    return intervals