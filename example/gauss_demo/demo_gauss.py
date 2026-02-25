"""
gauss_demo/demo_gauss.py
========================
TDA graph demo on a synthetic 2-D dataset sampled from three
multivariate normal distributions with different shapes.

Point cloud
-----------
  Blob A  (60 pts)  mean=(-3,  0)  elongated vertically  (sigma_x < sigma_y)
  Blob B  (60 pts)  mean=( 0,  2.5)  tilted ellipse
  Blob C  (60 pts)  mean=( 3, -0.5)  circular

The data is already 2-D, so no dimensionality reduction is needed.

Pipeline (run twice: single linkage then complete linkage)
----------------------------------------------------------
  Step 1  hierarchical clustering with chosen linkage (absolute cutoff)
  Step 2  2-D square covering: equally distributed 4 x 4 grid
          of overlapping rectangular patches
  Step 3  TDA graph – edge (i,j) iff same cluster AND share a patch

The two linkage methods highlight how chaining affects single linkage:
blobs B and C (which are close) merge into one cluster under single
linkage, while complete linkage correctly recovers all three Gaussians.

Outputs (saved in this directory)
----------------------------------
  01_raw_data.png          scatter coloured by true Gaussian source
  02_covering.png          2-D grid cover (single-linkage clusters)
  03_clustering.png        single-linkage clustering panels
  04_tda_graph.png         TDA graph with single-linkage clusters
  05_complete_covering.png 2-D grid cover (complete-linkage clusters)
  06_complete_clustering.png  complete-linkage clustering panels
  07_complete_tda_graph.png   TDA graph with complete-linkage clusters
  08_cross_tda_graph.png      complete-linkage graph + cross-cluster proximity edges
                              (dashed orange: dist ≤ canonical_cut_ across cluster boundaries)
"""

import sys
import os

_HERE   = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from tda_graph import TDAGraph

# ── Configuration ───────────────────────────────────────────────────────────
N_INTERVALS = 4      # 4 x 4 = 16 square patches
OVERLAP     = 0.40   # 40 % extension on each side
TAU_ABS     = -1.0   # cut=1.0 gives exactly 60/60/60 (the three blobs)
SEED        = 7

BLOB_NAMES  = ["Blob A", "Blob B", "Blob C"]
BLOB_COLORS = ["#4878d0", "#e05c2a", "#2ca02c"]

# ── Generate point cloud ─────────────────────────────────────────────────────
print("=" * 60)
print("Synthetic Gaussians  --  TDA graph with 2-D square cover")
print("=" * 60)

rng = np.random.default_rng(SEED)

A = rng.multivariate_normal([-3.0,  0.0],
                             [[0.50, 0.10], [0.10, 0.80]], 60)  # tall ellipse
B = rng.multivariate_normal([ -0.7,  2],
                             [[0.50, -0.20], [-0.20, 0.25]], 60) # tilted ellipse
C = rng.multivariate_normal([ 0.6, -0.5],
                             [[0.10, 0.00], [0.00, 0.10]], 30)   # circular

X      = np.vstack([A, B, C])
y_true = np.array([0] * 60 + [1] * 60 + [2] * 30)  # ground-truth (unused by TDAGraph)

print(f"\nPoint cloud : {len(X)} points  "
      f"({sum(y_true==0)} Blob A, {sum(y_true==1)} Blob B, {sum(y_true==2)} Blob C)")

# ── Figure 1 – Raw data coloured by true label ───────────────────────────────
print("\nPlotting Figure 1 - raw data...")

fig1, ax1 = plt.subplots(figsize=(7, 5.5))
fig1.patch.set_facecolor("#f7f7f7")
ax1.set_facecolor("#fafafa")

for pts, name, col in zip([A, B, C], BLOB_NAMES, BLOB_COLORS):
    ax1.scatter(pts[:, 0], pts[:, 1], s=28, c=col,
                edgecolors="white", linewidths=0.4, alpha=0.88, label=name)

ax1.set_title(
    "Synthetic dataset  --  3 multivariate Gaussians in 2-D\n"
    "colour = true Gaussian source",
    fontsize=11, fontweight="bold",
)
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.legend(fontsize=9, loc="upper right")
ax1.set_aspect("equal")
ax1.grid(True, linewidth=0.3, alpha=0.4)
fig1.tight_layout()

_p1 = os.path.join(_HERE, "01_raw_data.png")
fig1.savefig(_p1, dpi=150, bbox_inches="tight", facecolor=fig1.get_facecolor())
print(f"  Saved: {_p1}")
plt.close(fig1)

# ── Fit TDAGraph ─────────────────────────────────────────────────────────────
print(f"\nFitting TDAGraph  (n_intervals={N_INTERVALS}x{N_INTERVALS}, "
      f"overlap={OVERLAP}, tau={TAU_ABS})...")

tda = TDAGraph(n_intervals=N_INTERVALS, overlap_frac=OVERLAP, tau=TAU_ABS)
tda.fit(X)

print(f"  Canonical cut : {tda.canonical_cut_:.4f}  (absolute)")
print(f"  Clusters      : {tda.n_clusters_}")
print()
tda.summary()

# ── Figure 2 – 2-D square cover ──────────────────────────────────────────────
print("\nPlotting Figure 2 - 2-D covering...")
_p2 = os.path.join(_HERE, "02_covering.png")
tda.plot_covering(figsize=(8, 6), save_path=_p2)
plt.close("all")

# ── Figure 3 – Clustering panels ─────────────────────────────────────────────
print("Plotting Figure 3 - clustering...")
_p3 = os.path.join(_HERE, "03_clustering.png")
# Sweep from fragmented (0.3) through the correct split (1.0) to merged (1.5)
ABS_CUTS = [0.3, 0.5, 0.7, 1.0, 1.2, 1.5]
tda.plot_clustering(abs_cuts=ABS_CUTS, figsize=(14, 9), save_path=_p3)
plt.close("all")

# ── Figure 4 – TDA graph ──────────────────────────────────────────────────────
print("Plotting Figure 4 - TDA graph...")
_p4 = os.path.join(_HERE, "04_tda_graph.png")
tda.plot_graph_graphviz(save_path=_p4)

print(f"\n-- Single-linkage outputs saved (01–04) --")

# ── Complete-linkage run ──────────────────────────────────────────────────────
# Complete linkage uses the *maximum* pairwise distance between clusters,
# which avoids the chaining effect: blobs B and C are correctly separated.
print("\n" + "=" * 60)
print("Complete-linkage run")
print("=" * 60)

tda_c = TDAGraph(
    n_intervals=N_INTERVALS,
    overlap_frac=OVERLAP,
    tau="auto",                  # abs-gap: finds the 3-cluster cut automatically
    linkage_method="complete",
)
tda_c.fit(X)

print(f"  Canonical cut : {tda_c.canonical_cut_:.4f}  (abs-gap, complete linkage)")
print(f"  Clusters      : {tda_c.n_clusters_}")
print()
tda_c.summary()

# ── Figure 5 – 2-D cover (complete linkage) ───────────────────────────────────
print("\nPlotting Figure 5 - 2-D covering (complete linkage)...")
_p5 = os.path.join(_HERE, "05_complete_covering.png")
tda_c.plot_covering(figsize=(8, 6), save_path=_p5)
plt.close("all")

# ── Figure 6 – Clustering panels (complete linkage) ──────────────────────────
print("Plotting Figure 6 - clustering (complete linkage)...")
_p6 = os.path.join(_HERE, "06_complete_clustering.png")
# Sweep covers fragmented (1.5) through correct 3-cluster cut (~3.6) to merged
COMPLETE_CUTS = [1.5, 2.5, 3.0, 3.6, 4.5, 6.0]
tda_c.plot_clustering(abs_cuts=COMPLETE_CUTS, figsize=(14, 9), save_path=_p6)
plt.close("all")

# ── Figure 7 – TDA graph (complete linkage) ───────────────────────────────────
print("Plotting Figure 7 - TDA graph (complete linkage)...")
_p7 = os.path.join(_HERE, "07_complete_tda_graph.png")
tda_c.plot_graph_graphviz(save_path=_p7)

print(f"\n-- Complete-linkage outputs saved (05–07) --")

# ── Cross-cluster proximity edges (complete linkage) ──────────────────────────
# Re-fit with cross_cluster_edges=True.  The canonical cut is the same as the
# complete-linkage run above; we now also add edges between points in the same
# patch whose Euclidean distance is ≤ canonical_cut_, even across cluster
# boundaries.  Under single linkage this would add nothing (same-cluster ↔
# within-cut-distance), but under complete linkage some near-boundary pairs
# from different clusters can still be within the cut distance.
print("\n" + "=" * 60)
print("Complete-linkage  +  cross-cluster proximity edges")
print("=" * 60)

tda_cx = TDAGraph(
    n_intervals=N_INTERVALS,
    overlap_frac=OVERLAP,
    tau="auto",
    linkage_method="complete",
    cross_cluster_edges=True,
)
tda_cx.fit(X)

print(f"  Canonical cut       : {tda_cx.canonical_cut_:.4f}  (complete linkage)")
print(f"  Clusters            : {tda_cx.n_clusters_}")
print(f"  Same-cluster edges  : {tda_cx.n_edges_}")
print(f"  Cross-cluster edges : {tda_cx.n_cross_edges_}")
print()
tda_cx.summary()

# ── Figure 8 – TDA graph with cross-cluster proximity edges ───────────────────
print("\nPlotting Figure 8 - TDA graph with cross-cluster proximity edges...")
_p8 = os.path.join(_HERE, "08_cross_tda_graph.png")
tda_cx.plot_graph_graphviz(save_path=_p8)

print(f"\nAll outputs saved to: {_HERE}")
