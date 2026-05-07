"""
mnist_demo/demo_mnist.py
========================
TDA graph demo on a 2-D point cloud derived from MNIST digit "9".
"""

import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_t_start = time.time()


def _elapsed() -> str:
    return f"{time.time() - _t_start:.2f}s"


def _step(msg: str) -> None:
    print(f"\n[{_elapsed()}]  {msg}", flush=True)


_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

_step("Importing TDAGraph...")
from tda_graph import TDAGraph

_step("TDAGraph import OK")

# Configuration
N_INTERVALS = 5
OVERLAP = 0.40
N_POINTS = 80
NOISE_STD = 0.3
SEED = 42
TARGET_DIGIT = 9


def _load_mnist_digit_samples(target_digit: int) -> np.ndarray:
    """
    Return all images of the requested digit as shape (n, 28, 28), float.

    Local file is preferred (offline and deterministic). OpenML is fallback.
    """
    local_path = os.path.join(_HERE, "mnist.npz")
    if os.path.exists(local_path):
        _step(f"Loading local MNIST file: {local_path}")
        with np.load(local_path) as data:
            x_train = data["x_train"].astype(float)
            y_train = data["y_train"].astype(int)
        subset = x_train[y_train == int(target_digit)]
        _step(f"Local MNIST load OK: {len(subset)} samples of digit '{target_digit}'")
        return subset

    _step("Local mnist.npz not found. Falling back to sklearn.fetch_openml...")
    from sklearn.datasets import fetch_openml

    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    x_all = mnist.data.reshape(-1, 28, 28).astype(float)
    y_all = mnist.target.astype(str)
    subset = x_all[y_all == str(target_digit)]
    _step(f"OpenML load OK: {len(subset)} samples of digit '{target_digit}'")
    return subset


def _select_canonical_digit(samples: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Pick the sample closest to the class mean in L2 distance.
    This usually gives a more "typical" looking digit than max-ink selection.
    """
    flat = samples.reshape(samples.shape[0], -1)
    mean_vec = flat.mean(axis=0, keepdims=True)
    d2 = np.sum((flat - mean_vec) ** 2, axis=1)
    idx = int(np.argmin(d2))
    return samples[idx], idx


print("=" * 60)
print("MNIST digit 9  --  TDA graph with 2-D square cover")
print("=" * 60)

nines = _load_mnist_digit_samples(TARGET_DIGIT)

_step("Selecting a canonical digit-9 image and sampling points...")
img, best_idx = _select_canonical_digit(nines)
rows, cols = np.where(img > 0)
pix = img[rows, cols]

rng = np.random.default_rng(SEED)
if len(rows) > N_POINTS:
    # Keep brightest pixels so the sampled cloud preserves the digit silhouette.
    keep = np.argsort(-pix)[:N_POINTS]
    rows = rows[keep]
    cols = cols[keep]

X_clean = np.column_stack([cols.astype(float), -rows.astype(float)])
X = X_clean + rng.normal(0.0, NOISE_STD, size=X_clean.shape)
_step(f"Point cloud built: {len(X)} points (sample #{best_idx}, noise={NOISE_STD})")

_step("Plotting Figure 1 - raw data...")
fig1, ax1 = plt.subplots(figsize=(5, 7))
fig1.patch.set_facecolor("#f7f7f7")
ax1.set_facecolor("#fafafa")

ax1.scatter(
    X[:, 0], X[:, 1], color="#2b6cb0", s=28, edgecolors="white", linewidths=0.3, alpha=0.90
)

ax1.set_title(
    "MNIST digit '9' - sampled 2-D shape point cloud\n"
    f"({len(X)} points from one digit image, noise={NOISE_STD})",
    fontsize=11,
    fontweight="bold",
)
ax1.set_xlabel("x (pixel column)")
ax1.set_ylabel("y (-pixel row)")
ax1.set_aspect("equal")
ax1.grid(True, linewidth=0.3, alpha=0.4)
fig1.tight_layout()

_p1 = os.path.join(_HERE, "01_raw_data.png")
fig1.savefig(_p1, dpi=150, bbox_inches="tight", facecolor=fig1.get_facecolor())
plt.close(fig1)
_step(f"Saved: {_p1}")

print("\n" + "=" * 60)
print("Hausdorff-linkage run")
print("=" * 60)

_step(f"Fitting TDAGraph (hausdorff, {len(X)} pts)...")
tda_h = TDAGraph(
    n_intervals=N_INTERVALS,
    overlap_frac=OVERLAP,
    tau="auto",
    linkage_method="hausdorff",
)
tda_h.fit(X)
_step(f"Fit done: cut={tda_h.canonical_cut_:.4f}, clusters={tda_h.n_clusters_}")

_p2 = os.path.join(_HERE, "02_hausdorff_covering.png")
tda_h.plot_covering(figsize=(8, 9), save_path=_p2)
plt.close("all")
_step(f"Saved: {_p2}")

_p3 = os.path.join(_HERE, "03_hausdorff_clustering.png")
tda_h.plot_clustering(figsize=(14, 9), save_path=_p3)
plt.close("all")
_step(f"Saved: {_p3}")

_p4 = os.path.join(_HERE, "04_hausdorff_rel_jumps.png")
tda_h.plot_rel_jumps(figsize=(10, 4), save_path=_p4)
plt.close("all")
_step(f"Saved: {_p4}")

_p5 = os.path.join(_HERE, "05_hausdorff_tda_graph")
tda_h.plot_graph_graphviz(engine="neato", save_path=_p5, fmt="png")
_step(f"Saved: {_p5}.png")

print("\n" + "=" * 60)
print("Single-linkage run")
print("=" * 60)

_step(f"Fitting TDAGraph (single, {len(X)} pts)...")
tda_s = TDAGraph(
    n_intervals=N_INTERVALS,
    overlap_frac=OVERLAP,
    tau="auto",
    linkage_method="single",
)
tda_s.fit(X)
_step(f"Fit done: cut={tda_s.canonical_cut_:.4f}, clusters={tda_s.n_clusters_}")

_p6 = os.path.join(_HERE, "06_single_covering.png")
tda_s.plot_covering(figsize=(8, 9), save_path=_p6)
plt.close("all")
_step(f"Saved: {_p6}")

_p7 = os.path.join(_HERE, "07_single_clustering.png")
tda_s.plot_clustering(figsize=(14, 9), save_path=_p7)
plt.close("all")
_step(f"Saved: {_p7}")

_p8 = os.path.join(_HERE, "08_single_rel_jumps.png")
tda_s.plot_rel_jumps(figsize=(10, 4), save_path=_p8)
plt.close("all")
_step(f"Saved: {_p8}")

_p9 = os.path.join(_HERE, "09_single_tda_graph")
tda_s.plot_graph_graphviz(engine="neato", save_path=_p9, fmt="png")
_step(f"Saved: {_p9}.png")

_step(f"DONE - all outputs saved to: {_HERE}")
