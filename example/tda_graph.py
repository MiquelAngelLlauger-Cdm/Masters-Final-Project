"""
tda_graph.py
============
TDAGraph – a TDA-inspired proximity graph built from single-linkage
clustering and a Mapper-style 2-D cover.

Pipeline
--------
1. **Clustering** – build a single-linkage dendrogram; cut it using a
   relative-ratio cutoff  r_i = h[i+1] / h[i].
2. **Covering**   – partition the 2-D bounding box into an N×N grid of
   overlapping rectangular patches.
3. **Graph**      – add an undirected edge (i, j) iff
                    (a) i and j share the same cluster label, AND
                    (b) at least one cover patch contains both.

Usage
-----
    from tda_graph import TDAGraph
    from sklearn.datasets import make_moons

    X, _ = make_moons(n_samples=200, noise=0.08, random_state=0)

    tda = TDAGraph(n_intervals=5, overlap_frac=0.4, tau="auto")
    tda.fit(X)
    tda.summary()
    tda.plot_pipeline(save_prefix="my_run")
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from scipy.cluster.hierarchy import linkage, fcluster


# ──────────────────────────────────────────────────────────────────────────────

class TDAGraph:
    """
    TDA proximity graph from single-linkage clustering + Mapper-style cover.

    Parameters
    ----------
    n_intervals : int, default 5
        Number of grid cells per axis in the 2-D cover (N × N patches total).
    overlap_frac : float, default 0.4
        Fractional extension on each side of each base interval.
        E.g. 0.4 → each patch is 40 % wider on each side than a base cell.
    tau : float or ``"auto"``, default ``"auto"``
        Controls where the dendrogram is cut.

        * **"auto"**    – cut at the step with the *maximum* relative jump
          r_i = h[i+1]/h[i], skipping early steps where h[i] < 5 % of the
          max merge distance (those ratios are inflated by near-zero denominators).
        * **float > 1** – first-jump criterion: cut at the *first* step where
          r_i > tau.

    linkage_method : str, default ``"single"``
        How cluster-to-cluster distances are computed at each merge step.

        * Standard scipy methods: ``"single"``, ``"complete"``,
          ``"average"``, ``"ward"``.
        * ``"hausdorff"`` – custom greedy linkage using the Hausdorff
          distance between clusters:
          ``d_H(A,B) = max(max_{a∈A} min_{b∈B} d(a,b),
          max_{b∈B} min_{a∈A} d(a,b))``.
          Stricter than single-linkage, less aggressive than
          complete-linkage; reflects the worst-case nearest-neighbour
          proximity between the two point clouds.

    Attributes (set by :meth:`fit`)
    ---------------------------------
    X_              : (n, 2) ndarray   – input data
    Z_              : (n-1, 4) ndarray – scipy linkage matrix
    heights_        : (n-1,) ndarray   – sorted merge distances
    rel_jumps_      : (n-2,) ndarray   – r_i = h[i+1] / h[i]
    canonical_tau_  : float            – tau used for the canonical clustering
    canonical_cut_  : float            – absolute distance threshold
    labels_         : (n,) int ndarray – cluster labels (1-indexed)
    patches_        : list[(x0,x1,y0,y1)] – cover patch bounds
    membership_     : list[ndarray]    – point indices per patch
    edges_          : set[(i,j)]       – undirected edges (i < j)
    """

    # ── Construction ──────────────────────────────────────────────────────────

    def __init__(
        self,
        n_intervals: int = 5,
        overlap_frac: float = 0.4,
        tau: float | str = "auto",
        linkage_method: str = "single",
    ) -> None:
        self.n_intervals         = n_intervals
        self.overlap_frac        = overlap_frac
        self.tau            = tau
        self.linkage_method = linkage_method

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, lens: np.ndarray | None = None) -> "TDAGraph":
        """
        Fit the full pipeline to a 2-D point cloud.

        Parameters
        ----------
        X : array-like of shape (n_samples, 2)
            2-D point cloud used for clustering and as graph node positions.
        lens : array-like of shape (n_samples,), optional
            1-D lens (filter) values, one per point.  When provided the cover
            is built as ``n_intervals`` overlapping intervals on the lens axis
            instead of an ``n_intervals × n_intervals`` 2-D grid.
            A common choice is ``lens = X[:, 1]`` (projection onto the y-axis).

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != 2:
            raise ValueError(
                f"X must have shape (n_samples, 2); got {X.shape}."
            )
        if X.shape[0] < 3:
            raise ValueError("Need at least 3 samples.")

        self.X_    = X
        self.lens_ = None if lens is None else np.asarray(lens, dtype=float).ravel()

        self._fit_clustering(self.tau)
        self._fit_covering()
        self._fit_graph()
        return self

    # ── Private step methods ──────────────────────────────────────────────────

    def _fit_clustering(self, tau: float | str) -> None:
        """Step 1 – dendrogram + cut.

        Two cut modes controlled by ``tau``:

        * ``"auto"``    – cut at the step with the largest relative jump
                          r_i = h[i+1] / h[i], ignoring initial steps where
                          h[i] < 5 % of the max merge distance (those early
                          ratios are inflated by a near-zero denominator).
        * ``float > 1`` – **first-jump**: cut at the first step where
                          h[i+1]/h[i] > tau.
        """
        if self.linkage_method == "hausdorff":
            self.Z_ = self._hausdorff_linkage()
        else:
            self.Z_ = linkage(self.X_, method=self.linkage_method, metric="euclidean")
        self.heights_   = self.Z_[:, 2]                        # ascending
        self.rel_jumps_ = self.heights_[1:] / self.heights_[:-1]

        if tau == "auto":
            rj    = self.rel_jumps_
            h     = self.heights_
            # Skip early steps whose denominator h[i] is still near zero —
            # those produce artificially large ratios that obscure the real gap.
            min_h = float(h[-1]) * 0.05
            valid = np.where(h[:-1] >= min_h)[0]   # h[:-1] are denominators of rj
            i_max = int(valid[np.argmax(rj[valid])]) if len(valid) else int(np.argmax(rj))
            self.canonical_cut_ = float(h[i_max])
            self.canonical_tau_ = float(rj[i_max])
        else:
            self.canonical_tau_ = float(tau)
            self.canonical_cut_ = self._cutoff_distance(self.canonical_tau_)

        self.labels_ = fcluster(
            self.Z_, t=self.canonical_cut_, criterion="distance"
        )

    def _hausdorff_linkage(self) -> np.ndarray:
        """
        Build a scipy-compatible linkage matrix using Hausdorff distance.

        Greedy O(n³) algorithm: at each step merge the pair of active
        clusters (A, B) that minimises

            d_H(A,B) = max( max_{a∈A} min_{b∈B} d(a,b),
                            max_{b∈B} min_{a∈A} d(a,b) )

        The full pairwise Euclidean distance matrix D is precomputed once
        and reused via index subsets, so no point coordinates need to be
        recalculated during the merge loop.

        Returns
        -------
        Z : (n-1, 4) ndarray
            Scipy linkage matrix columns: [id_a, id_b, distance, count].
        """
        from scipy.spatial.distance import cdist

        X  = self.X_
        n  = len(X)
        D  = cdist(X, X, metric="euclidean")

        # clusters[id] = list of original point indices
        clusters = {i: [i] for i in range(n)}
        active   = list(range(n))

        Z       = np.zeros((n - 1, 4))
        next_id = n

        for step in range(n - 1):
            best_h    = np.inf
            best_pair = (active[0], active[1])

            for ai in range(len(active)):
                ca = active[ai]
                pa = clusters[ca]
                for bi in range(ai + 1, len(active)):
                    cb  = active[bi]
                    pb  = clusters[cb]
                    sub = D[np.ix_(pa, pb)]
                    # directed Hausdorff: max over each side of min to the other
                    h = max(sub.min(axis=1).max(), sub.min(axis=0).max())
                    if h < best_h:
                        best_h    = h
                        best_pair = (ca, cb)

            ca, cb   = best_pair
            new_id   = next_id
            next_id += 1
            merged   = clusters[ca] + clusters[cb]

            Z[step] = [ca, cb, best_h, len(merged)]

            clusters[new_id] = merged
            active.remove(ca)
            active.remove(cb)
            active.append(new_id)
            del clusters[ca]
            del clusters[cb]

        return Z

    def _fit_covering(self) -> None:
        """Step 2 – dispatch to 1-D lens cover or 2-D grid cover."""
        if getattr(self, "lens_", None) is not None:
            self._fit_covering_1d()
        else:
            self._fit_covering_2d()

    def _fit_covering_2d(self) -> None:
        """N×N overlapping grid cover of the 2-D bounding box."""
        X   = self.X_
        pad = 0.02
        xr  = X[:, 0].max() - X[:, 0].min()
        yr  = X[:, 1].max() - X[:, 1].min()
        x_lo = X[:, 0].min() - pad * xr
        x_hi = X[:, 0].max() + pad * xr
        y_lo = X[:, 1].min() - pad * yr
        y_hi = X[:, 1].max() + pad * yr

        n  = self.n_intervals
        ov = self.overlap_frac
        sx, sy = (x_hi - x_lo) / n, (y_hi - y_lo) / n
        ex, ey = sx * ov / 2, sy * ov / 2

        patches, membership = [], []
        for i in range(n):
            for j in range(n):
                px0, px1 = x_lo + i * sx - ex, x_lo + (i + 1) * sx + ex
                py0, py1 = y_lo + j * sy - ey, y_lo + (j + 1) * sy + ey
                inside = np.where(
                    (X[:, 0] >= px0) & (X[:, 0] <= px1) &
                    (X[:, 1] >= py0) & (X[:, 1] <= py1)
                )[0]
                patches.append((px0, px1, py0, py1))
                membership.append(inside)

        self.patches_    = patches
        self.membership_ = membership

    def _fit_covering_1d(self) -> None:
        """N overlapping intervals along the 1-D lens axis."""
        lens = self.lens_
        pad  = 0.02
        rng  = float(lens.max() - lens.min())
        lo   = float(lens.min()) - pad * rng
        hi   = float(lens.max()) + pad * rng

        n    = self.n_intervals
        ov   = self.overlap_frac
        step = (hi - lo) / n
        ext  = step * ov / 2

        patches, membership = [], []
        for i in range(n):
            a      = lo + i * step - ext
            b      = lo + (i + 1) * step + ext
            inside = np.where((lens >= a) & (lens <= b))[0]
            patches.append((a, b))
            membership.append(inside)

        self.patches_    = patches
        self.membership_ = membership

    def _fit_graph(self) -> None:
        """Step 3 – build the ε-neighbourhood graph filtered by the cover.

        For each cover patch, consider all point pairs (i, j) contained in it.
        Add edge (i, j) iff their Euclidean distance is at most ε = canonical_cut_
        (the merge distance at the largest relative jump of the dendrogram).
        Pairs that do not share any patch are discarded regardless of distance.
        """
        from scipy.spatial.distance import cdist

        eps = self.canonical_cut_
        D   = cdist(self.X_, self.X_, metric="euclidean")

        edges: set[tuple[int, int]] = set()
        for members in self.membership_:
            if len(members) < 2:
                continue
            for a in range(len(members)):
                for b in range(a + 1, len(members)):
                    i, j = int(members[a]), int(members[b])
                    if D[i, j] <= eps:
                        edges.add((min(i, j), max(i, j)))

        self.edges_ = edges

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def n_nodes_(self) -> int:
        """Number of nodes (= number of data points)."""
        return len(self.X_)

    @property
    def n_edges_(self) -> int:
        """Number of undirected edges in the TDA graph."""
        return len(self.edges_)

    @property
    def n_clusters_(self) -> int:
        """Number of clusters in the canonical clustering."""
        return int(len(np.unique(self.labels_)))

    @property
    def n_patches_(self) -> int:
        """Total number of cover patches (= n_intervals²)."""
        return len(self.patches_)

    @property
    def adjacency_matrix_(self):
        """
        Sparse CSR adjacency matrix of the TDA graph.

        Returns
        -------
        scipy.sparse.csr_matrix of shape (n, n), dtype uint8
        """
        from scipy.sparse import lil_matrix
        n = self.n_nodes_
        A = lil_matrix((n, n), dtype=np.uint8)
        for i, j in self.edges_:
            A[i, j] = A[j, i] = 1
        return A.tocsr()

    # ── Tau utilities ─────────────────────────────────────────────────────────

    def _cutoff_distance(self, tau: float) -> float:
        """Absolute cut distance: first step where r_i > tau."""
        idx = np.where(self.rel_jumps_ > tau)[0]
        return (
            float(self.heights_[idx[0]])
            if len(idx)
            else float(self.heights_[-1]) + 1e-10
        )

    def set_tau(self, tau: float | str) -> "TDAGraph":
        """
        Re-cluster with a new tau value and rebuild the graph.

        The cover is **not** recomputed. Useful for quickly exploring
        different clustering granularities on a fixed cover.

        Parameters
        ----------
        tau : float or "auto"

        Returns
        -------
        self
        """
        self._check_fitted()
        self.tau = tau
        self._fit_clustering(tau)
        self._fit_graph()
        return self

    def scan_tau(self, n_points: int = 500) -> tuple[np.ndarray, np.ndarray]:
        """
        Sweep tau in log-space and record the resulting cluster count.

        Useful for understanding the cluster-count landscape before
        choosing a canonical tau.

        Parameters
        ----------
        n_points : int
            Number of tau values to sample.

        Returns
        -------
        taus       : (n_points,) ndarray
        n_clusters : (n_points,) int ndarray
        """
        self._check_fitted()
        max_rj = float(self.rel_jumps_.max())
        if max_rj <= 1.0:
            return np.array([1.001]), np.array([1])

        taus = np.logspace(
            np.log10(1.0001), np.log10(max_rj * 0.9999), n_points
        )
        ncs = []
        for t in taus:
            d  = self._cutoff_distance(t)
            nc = int(len(np.unique(fcluster(self.Z_, t=d, criterion="distance"))))
            ncs.append(nc)
        return taus, np.array(ncs)

    def auto_taus(self, n_taus: int = 6) -> list[float]:
        """
        Automatically select ``n_taus`` thresholds that each correspond to
        a *different* cluster count, spread across the full landscape.

        Parameters
        ----------
        n_taus : int

        Returns
        -------
        list of float, sorted ascending
        """
        self._check_fitted()
        scan_taus, scan_ncs = self.scan_tau()

        # Collect one tau per cluster-count transition
        transitions, prev_nc = [], None
        for tau, nc in zip(scan_taus, scan_ncs):
            if nc != prev_nc:
                transitions.append(tau)
                prev_nc = nc

        n_t = len(transitions)
        if n_t == 0:
            return [float(self.canonical_tau_)] * n_taus

        if n_t >= n_taus:
            sel = np.round(np.linspace(0, n_t - 1, n_taus)).astype(int)
            return sorted(transitions[i] for i in sel)

        # Fewer transitions than requested: pad with nearby values
        while len(transitions) < n_taus:
            transitions.append(transitions[-1] * 1.01)
        return sorted(transitions[:n_taus])

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> None:
        """Print a human-readable summary of the fitted graph."""
        self._check_fitted()
        n_nonempty = sum(1 for m in self.membership_ if len(m) > 0)
        counts     = np.bincount(self.labels_)[1:]  # labels are 1-indexed

        print("TDAGraph summary")
        print(f"  Dataset    : {self.n_nodes_} points")
        print(f"  Clustering : {self.linkage_method}-linkage, "
              f"tau={self.canonical_tau_:.6f}, "
              f"cut={self.canonical_cut_:.6f}, "
              f"{self.n_clusters_} cluster(s)")
        max_shown = 20
        for c, cnt in enumerate(counts[:max_shown], start=1):
            print(f"               cluster {c}: {cnt} points")
        if len(counts) > max_shown:
            print(f"               ... ({len(counts) - max_shown} more clusters not shown)")
        if getattr(self, "lens_", None) is not None:
            print(f"  Cover      : {self.n_intervals} intervals on 1-D lens, "
                  f"overlap={self.overlap_frac:.0%}  "
                  f"({n_nonempty}/{self.n_patches_} non-empty)")
        else:
            print(f"  Cover      : {self.n_intervals}x{self.n_intervals} grid, "
                  f"overlap={self.overlap_frac:.0%}  "
                  f"({n_nonempty}/{self.n_patches_} patches non-empty)")
        print(f"  Graph      : {self.n_nodes_} nodes, {self.n_edges_} edges  "
              f"(ε={self.canonical_cut_:.4f})")

    # ── Visualisation ─────────────────────────────────────────────────────────

    def plot_clustering(
        self,
        taus: list[float] | None = None,
        abs_cuts: list[float] | None = None,
        n_taus: int = 6,
        figsize: tuple[float, float] = (14, 9),
        save_path: str | None = None,
    ):
        """
        Scatter plots for the canonical clustering plus multiple threshold values.

        The **first panel** always shows the canonical clustering.
        The remaining panels show different cut values:

        * If ``abs_cuts`` is provided: panels use those explicit absolute
          distance thresholds  (``fcluster(Z, t=d)``).
        * Otherwise: panels use the first-jump relative-ratio criterion
          ``r_i = h[i+1]/h[i] > tau`` swept over ``n_taus - 1`` values.

        Parameters
        ----------
        taus      : list of float, optional
            Explicit tau values for the first-jump panels.
        abs_cuts  : list of float, optional
            Explicit absolute distance thresholds.  When given, ``taus`` and
            ``n_taus`` are ignored for the non-canonical panels.
        n_taus    : int
            Total number of panels (including the canonical one).
        figsize   : tuple
        save_path : str, optional

        Returns
        -------
        fig, axes
        """
        self._check_fitted()

        use_abs = abs_cuts is not None
        if use_abs:
            extra_vals = list(abs_cuts)
        elif taus is not None:
            extra_vals = list(taus)
        else:
            extra_vals = self.auto_taus(max(1, n_taus - 1))

        n     = len(extra_vals) + 1        # +1 for the canonical panel
        ncols = min(3, n)
        nrows = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        fig.patch.set_facecolor("#f7f7f7")

        if use_abs:
            other_desc = "Other panels: absolute distance cutoffs"
        else:
            other_desc = r"Other panels: first-jump criterion  $r_i = h_{i+1}/h_i > \tau$"

        # Canonical cut description depends on how it was set
        if self.tau == "auto":
            can_desc = f"Canonical (abs-gap heuristic): {self.n_clusters_} cluster(s)"
        elif isinstance(self.tau, (int, float)) and float(self.tau) < 0:
            can_desc = f"Canonical (absolute cut={self.canonical_cut_:.4f}): {self.n_clusters_} cluster(s)"
        else:
            can_desc = f"Canonical (first-jump tau={self.canonical_tau_:.4f}): {self.n_clusters_} cluster(s)"

        fig.suptitle(
            f"Step 1 – Single-Linkage Clustering\n"
            f"{self.n_nodes_} points  |  {can_desc}  |  {other_desc}",
            fontsize=11, fontweight="bold",
        )

        cmap    = plt.get_cmap("tab10")
        ax_list = list(axes.flat)

        # ── Panel 0: canonical cut ─────────────────────────────────────────────
        ax0     = ax_list[0]
        colors0 = self._cluster_colors(self.labels_, cmap)
        ax0.scatter(self.X_[:, 0], self.X_[:, 1], c=colors0, s=20,
                    edgecolors="white", linewidths=0.3, alpha=0.9)
        ax0.set_title(
            can_desc + f"\ncut={self.canonical_cut_:.4f}",
            fontsize=9.5, fontweight="bold",
        )
        ax0.set_facecolor("#fff7e6")   # warm tint to distinguish canonical panel
        for spine in ax0.spines.values():
            spine.set_edgecolor("#e05c2a")
            spine.set_linewidth(1.5)
        ax0.grid(True, linewidth=0.3, alpha=0.4)
        ax0.tick_params(labelsize=7)
        ax0.set_xlabel("x", fontsize=8)
        ax0.set_ylabel("y", fontsize=8)

        # ── Panels 1…: extra cut values ────────────────────────────────────────
        for ax, val in zip(ax_list[1:], extra_vals):
            if use_abs:
                d      = float(val)
                labels = fcluster(self.Z_, t=d, criterion="distance")
                nc     = int(len(np.unique(labels)))
                title  = f"cut={d:.4f}  (absolute)  |  {nc} cluster{'s' if nc > 1 else ''}"
            else:
                d      = self._cutoff_distance(float(val))
                labels = fcluster(self.Z_, t=d, criterion="distance")
                nc     = int(len(np.unique(labels)))
                title  = f"tau={val:.4f}  |  cut={d:.4f}  |  {nc} cluster{'s' if nc > 1 else ''}"
            colors = self._cluster_colors(labels, cmap)

            ax.scatter(self.X_[:, 0], self.X_[:, 1], c=colors, s=20,
                       edgecolors="white", linewidths=0.3, alpha=0.9)
            ax.set_title(title, fontsize=9.5, fontweight="bold")
            ax.set_facecolor("#fafafa")
            ax.grid(True, linewidth=0.3, alpha=0.4)
            ax.tick_params(labelsize=7)
            ax.set_xlabel("x", fontsize=8)
            ax.set_ylabel("y", fontsize=8)

        for ax in ax_list[n:]:
            ax.set_visible(False)

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            print(f"Saved: {save_path}")
        return fig, axes

    def plot_covering(
        self,
        figsize: tuple[float, float] = (9, 6.5),
        save_path: str | None = None,
    ):
        """
        Visualise the cover overlaid on the data (Step 2).

        For a **2-D grid** cover: rectangular patches are drawn over the
        scatter plot.
        For a **1-D lens** cover: overlapping horizontal bands are drawn,
        one per interval, together with a marginal lens-value distribution.

        Returns
        -------
        fig, ax  (or fig, (ax_main, ax_hist) for 1-D cover)
        """
        self._check_fitted()
        if getattr(self, "lens_", None) is not None:
            return self._plot_covering_1d(figsize, save_path)
        return self._plot_covering_2d(figsize, save_path)

    def _plot_covering_2d(
        self,
        figsize: tuple[float, float] = (9, 6.5),
        save_path: str | None = None,
    ):
        """Internal: 2-D grid covering visualisation."""
        cmap_cover   = plt.get_cmap("tab20b")
        cmap_cluster = plt.get_cmap("tab10")

        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("#f7f7f7")
        ax.set_facecolor("#fafafa")

        for k, ((px0, px1, py0, py1), members) in enumerate(
                zip(self.patches_, self.membership_)):
            col  = cmap_cover(k / self.n_patches_)
            rect = mpatches.Rectangle(
                (px0, py0), px1 - px0, py1 - py0,
                linewidth=1.0,
                edgecolor=(*col[:3], 0.65),
                facecolor=(*col[:3], 0.11 if len(members) > 0 else 0.03),
            )
            ax.add_patch(rect)
            if len(members) > 0:
                ax.text(
                    (px0 + px1) / 2, (py0 + py1) / 2,
                    f"{k}\n({len(members)})",
                    ha="center", va="center", fontsize=5.5,
                    color=(*col[:3], 0.80),
                )

        pt_colors = self._cluster_colors(self.labels_, cmap_cluster)
        ax.scatter(self.X_[:, 0], self.X_[:, 1], c=pt_colors, s=28,
                   edgecolors="white", linewidths=0.4, zorder=5, alpha=0.92)
        ax.legend(handles=self._legend_handles(cmap_cluster),
                  fontsize=9, loc="upper right")
        ax.set_title(
            f"Step 2 – 2-D Mapper-style Covering  "
            f"({self.n_intervals}x{self.n_intervals} grid, "
            f"overlap={self.overlap_frac:.0%})\n"
            f"Points coloured by cluster  ({self.n_clusters_} cluster(s))  "
            f"|  patch label = index (count)",
            fontsize=11, fontweight="bold",
        )
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.grid(True, linewidth=0.3, alpha=0.4)
        ax.autoscale()

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            print(f"Saved: {save_path}")
        return fig, ax

    def _plot_covering_1d(
        self,
        figsize: tuple[float, float] = (12, 7),
        save_path: str | None = None,
    ):
        """Internal: 1-D lens covering visualisation.

        Left panel  – 2-D scatter coloured by cluster label; horizontal
                      semi-transparent bands show the lens intervals.
        Right panel – histogram of lens values with the same interval bands
                      overlaid, so the overlap structure is visible.
        """
        from matplotlib.colors import Normalize
        cmap_cluster = plt.get_cmap("tab10")
        cmap_band    = plt.get_cmap("tab20b")

        fig, (ax_sc, ax_hist) = plt.subplots(
            1, 2, figsize=figsize,
            gridspec_kw={"width_ratios": [2, 1]},
        )
        fig.patch.set_facecolor("#f7f7f7")

        x_lo = float(self.X_[:, 0].min())
        x_hi = float(self.X_[:, 0].max())

        # ── left: scatter + horizontal interval bands ──────────────────────
        ax_sc.set_facecolor("#fafafa")
        for k, ((a, b), members) in enumerate(
                zip(self.patches_, self.membership_)):
            col  = cmap_band(k / self.n_patches_)
            rect = mpatches.Rectangle(
                (x_lo, a), x_hi - x_lo, b - a,
                linewidth=0.8,
                edgecolor=(*col[:3], 0.55),
                facecolor=(*col[:3], 0.10 if len(members) > 0 else 0.02),
                zorder=1,
            )
            ax_sc.add_patch(rect)
            if len(members) > 0:
                ax_sc.text(
                    x_hi, (a + b) / 2,
                    f" {k}({len(members)})",
                    va="center", ha="left", fontsize=6,
                    color=(*col[:3], 0.75),
                )

        pt_colors = self._cluster_colors(self.labels_, cmap_cluster)
        ax_sc.scatter(self.X_[:, 0], self.X_[:, 1],
                      c=pt_colors, s=20, edgecolors="white",
                      linewidths=0.3, zorder=3, alpha=0.92)
        ax_sc.legend(handles=self._legend_handles(cmap_cluster),
                     fontsize=8, loc="upper right")
        ax_sc.set_title(
            f"Step 2 – 1-D Lens Cover  "
            f"({self.n_intervals} intervals, overlap={self.overlap_frac:.0%})\n"
            f"Horizontal bands = lens intervals  |  "
            f"Points coloured by cluster ({self.n_clusters_})",
            fontsize=10, fontweight="bold",
        )
        ax_sc.set_xlabel("Feature 1")
        ax_sc.set_ylabel("Feature 2 (lens axis)")
        ax_sc.grid(True, linewidth=0.3, alpha=0.4)
        ax_sc.autoscale()

        # ── right: lens histogram + interval shading ───────────────────────
        ax_hist.set_facecolor("#fafafa")
        ax_hist.hist(self.lens_, bins=40, orientation="horizontal",
                     color="#4878d0", alpha=0.55, edgecolor="white",
                     linewidth=0.4)
        for k, (a, b) in enumerate(self.patches_):
            col = cmap_band(k / self.n_patches_)
            ax_hist.axhspan(a, b, alpha=0.18, color=col[:3], zorder=0)
            ax_hist.axhline(a, color=(*col[:3], 0.45),
                            linewidth=0.7, linestyle="--")
        ax_hist.set_title("Lens distribution\n& intervals", fontsize=10,
                          fontweight="bold")
        ax_hist.set_xlabel("Count")
        ax_hist.set_ylabel("")
        ax_hist.tick_params(left=False, labelleft=False)
        ax_hist.grid(True, linewidth=0.3, alpha=0.4)

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            print(f"Saved: {save_path}")
        return fig, (ax_sc, ax_hist)

    def plot_graph(
        self,
        figsize: tuple[float, float] = (10, 7),
        save_path: str | None = None,
    ):
        """
        Visualise the TDA proximity graph (Step 3).

        Nodes are positioned at their 2-D data coordinates and coloured by
        cluster.  Edges are drawn as thin grey lines.

        Returns
        -------
        fig, ax
        """
        self._check_fitted()
        cmap = plt.get_cmap("tab10")

        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("#f7f7f7")
        ax.set_facecolor("#fafafa")

        if self.edges_:
            segments = [(self.X_[i], self.X_[j]) for i, j in self.edges_]
            lc = LineCollection(
                segments, linewidths=0.5, colors="#888888",
                alpha=0.40, zorder=1,
            )
            ax.add_collection(lc)

        pt_colors = self._cluster_colors(self.labels_, cmap)
        ax.scatter(self.X_[:, 0], self.X_[:, 1], c=pt_colors, s=32,
                   edgecolors="white", linewidths=0.5, zorder=3, alpha=0.95)

        handles = self._legend_handles(cmap)
        ax.legend(handles=handles, fontsize=9, loc="upper right")

        ax.set_title(
            "Step 3 – TDA Proximity Graph\n"
            r"(edge iff $d(i,j) \leq \varepsilon$ & same cover patch)  |  "
            f"{self.n_nodes_} nodes, {self.n_edges_} edges  |  "
            f"ε={self.canonical_cut_:.4f}  |  "
            f"{self.n_clusters_} cluster(s)  |  "
            f"{self.n_intervals}x{self.n_intervals} cover, "
            f"overlap={self.overlap_frac:.0%}",
            fontsize=11, fontweight="bold",
        )
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.grid(True, linewidth=0.3, alpha=0.4)
        ax.autoscale()

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            print(f"Saved: {save_path}")
        return fig, ax

    def plot_graph_graphviz(
        self,
        engine: str = "neato",
        save_path: str | None = None,
        fmt: str = "png",
    ):
        """
        Render the TDA proximity graph via Graphviz.

        Nodes are coloured by cluster.  When ``engine="neato"`` (default)
        nodes are pinned to their 2-D data coordinates via the ``pos``
        attribute (``!`` suffix), preserving the spatial layout.  Other
        engines (``"sfdp"``, ``"fdp"``, ``"dot"``) compute an automatic
        topology-based layout.

        Requires the ``graphviz`` Python package (``pip install graphviz``)
        **and** the Graphviz system binaries on PATH
        (https://graphviz.org/download/).

        Parameters
        ----------
        engine : str
            Graphviz layout engine.  ``"neato"`` preserves data coordinates;
            ``"sfdp"`` / ``"fdp"`` give a force-directed topology view.
        save_path : str, optional
            Output path.  The format extension is appended automatically if
            absent (e.g. ``"out/graph"`` → ``"out/graph.png"``).
            If *None* the graph is not written to disk.
        fmt : str
            Graphviz output format: ``"png"`` (default), ``"svg"``, ``"pdf"`` …

        Returns
        -------
        graphviz.Graph
        """
        try:
            import graphviz
        except ImportError as exc:
            raise ImportError(
                "The 'graphviz' Python package is required. "
                "Install with:  pip install graphviz\n"
                "Also ensure the Graphviz system binaries are on PATH: "
                "https://graphviz.org/download/"
            ) from exc

        from matplotlib.colors import to_hex as mpl_to_hex

        self._check_fitted()

        cmap  = plt.get_cmap("tab10")
        u     = np.unique(self.labels_)
        denom = max(len(u) - 1, 1)
        cluster_color = {c: mpl_to_hex(cmap((c - 1) / denom)) for c in u}

        dot = graphviz.Graph(engine=engine)
        title = (
            f"TDA ε-neighbourhood graph  |  "
            f"ε = {self.canonical_cut_:.4f}  |  "
            f"{self.n_nodes_} nodes, {self.n_edges_} edges  |  "
            f"{self.linkage_method}-linkage, {self.n_clusters_} cluster(s)"
        )
        dot.attr(bgcolor="#f7f7f7", pad="0.3", outputorder="edgesfirst",
                 label=title, labelloc="t", fontsize="11", fontname="Helvetica")
        if engine != "neato":
            dot.attr(overlap="prism", splines="true")
        dot.attr("node", shape="circle", style="filled",
                 width="0.12", fixedsize="true", label="", penwidth="0")
        dot.attr("edge", penwidth="0.4", color="#88888866")

        # ── Nodes ─────────────────────────────────────────────────────────
        X     = self.X_
        xr    = float(X[:, 0].max() - X[:, 0].min()) or 1.0
        yr    = float(X[:, 1].max() - X[:, 1].min()) or 1.0
        scale = 8.0  # canvas size in inches

        for i in range(len(X)):
            col   = cluster_color[int(self.labels_[i])]
            attrs = {"fillcolor": col}
            if engine == "neato":
                px = (X[i, 0] - X[:, 0].min()) / xr * scale
                py = (X[i, 1] - X[:, 1].min()) / yr * scale
                attrs["pos"] = f"{px:.4f},{py:.4f}!"   # ! pins the node
            dot.node(str(i), **attrs)

        # ── Edges ─────────────────────────────────────────────────────────
        for i, j in self.edges_:
            dot.edge(str(i), str(j))

        # ── Render ────────────────────────────────────────────────────────
        if save_path is not None:
            root = save_path
            if root.lower().endswith(f".{fmt}"):
                root = root[: -(len(fmt) + 1)]
            dot.render(root, format=fmt, cleanup=True)
            print(f"Saved: {root}.{fmt}")

        return dot

    def plot_rel_jumps(
        self,
        figsize: tuple[float, float] = (10, 4),
        save_path: str | None = None,
    ):
        """
        Plot the relative jumps r_i = h[i+1] / h[i] along the dendrogram.

        Each bar is one consecutive merge-distance ratio. A tall bar marks a
        structural gap — a natural cluster boundary. The canonical cut is
        highlighted with a vertical line and an annotation.

        Returns
        -------
        fig, (ax_ratio, ax_height) – ratio bar chart + merge-distance line
        """
        self._check_fitted()
        h  = self.heights_
        rj = self.rel_jumps_          # length n-2
        xs = np.arange(len(rj))       # step indices 0 … n-3

        # Step where the canonical cut falls: find the heights_ entry closest
        # to canonical_cut_ (exact match when cut was set by auto/first-jump).
        i_cut = int(np.argmin(np.abs(h - self.canonical_cut_)))

        # For first-jump tau, also mark where rj first exceeds canonical_tau_
        # (only meaningful when tau was given as a float > 1, not "auto" or abs).
        try:
            tau_val = float(self.canonical_tau_)
            is_nan  = np.isnan(tau_val)
        except (TypeError, ValueError):
            is_nan = True
        if not is_nan and tau_val > 1.0:
            fj_hits = np.where(rj > tau_val)[0]
            i_fj    = int(fj_hits[0]) if len(fj_hits) else None
        else:
            i_fj = None

        fig, (ax_r, ax_h) = plt.subplots(
            2, 1, figsize=figsize, sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )
        fig.patch.set_facecolor("#f7f7f7")

        # ── top: relative-jump bar chart ──────────────────────────────────────
        ax_r.set_facecolor("#fafafa")
        bar_colors = ["#4878d0"] * len(rj)
        bar_colors[i_cut] = "#e05c2a"          # highlight the canonical cut step
        ax_r.bar(xs, rj, color=bar_colors, width=0.8, alpha=0.80, zorder=2)
        ax_r.axvline(i_cut, color="#e05c2a", linewidth=1.5, linestyle="--", zorder=3,
                     label=f"canonical cut  (step {i_cut},  r={rj[i_cut]:.3f})")
        if i_fj is not None and i_fj != i_cut:
            ax_r.axvline(i_fj, color="#2ca02c", linewidth=1.5, linestyle=":",
                         zorder=3,
                         label=fr"first-jump $\tau$={self.canonical_tau_:.3f}  "
                               f"(step {i_fj},  r={rj[i_fj]:.3f})")
        ax_r.set_ylabel(r"$r_i = h_{i+1}\,/\,h_i$", fontsize=10)
        ax_r.set_title(
            fr"Relative jumps of the {self.linkage_method}-linkage dendrogram"
            r"  $r_i = h_{i+1}/h_i$"
            "\n"
            r"Orange bar = canonical cut.  "
            r"Blue bars = all other merge steps.",
            fontsize=11, fontweight="bold",
        )
        ax_r.legend(fontsize=9)
        ax_r.grid(True, axis="y", linewidth=0.3, alpha=0.4)
        ax_r.set_xlim(-0.5, len(rj) - 0.5)

        # ── bottom: merge-distance line ───────────────────────────────────────
        ax_h.set_facecolor("#fafafa")
        ax_h.plot(np.arange(len(h)), h, color="#555555", linewidth=1.2,
                  marker="o", markersize=3, zorder=2)
        ax_h.axvline(i_cut, color="#e05c2a", linewidth=1.5, linestyle="--", zorder=3)
        ax_h.axhline(self.canonical_cut_, color="#e05c2a", linewidth=1.0,
                     linestyle=":", alpha=0.7,
                     label=f"cut={self.canonical_cut_:.4f}")
        ax_h.set_ylabel(r"$h_i$", fontsize=10)
        ax_h.set_xlabel("Merge step  $i$", fontsize=10)
        ax_h.legend(fontsize=8)
        ax_h.grid(True, linewidth=0.3, alpha=0.4)

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            print(f"Saved: {save_path}")
        return fig, (ax_r, ax_h)

    def plot_tau_landscape(
        self,
        n_points: int = 500,
        figsize: tuple[float, float] = (9, 4),
        save_path: str | None = None,
    ):
        """
        Plot cluster count vs. tau (the cluster-count landscape).

        Also marks the canonical tau as a vertical line.

        Returns
        -------
        fig, ax
        """
        self._check_fitted()
        taus, ncs = self.scan_tau(n_points)

        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("#f7f7f7")
        ax.set_facecolor("#fafafa")

        ax.step(taus, ncs, where="post", color="#4878d0", linewidth=1.5,
                label=r"first-jump criterion  $r_i = h_{i+1}/h_i > \tau$")
        ax.axhline(
            self.n_clusters_, color="#e05c2a", linewidth=1.5, linestyle="--",
            label=f"canonical cut (max rel-jump) → {self.n_clusters_} cluster(s)"
                  f",  cut={self.canonical_cut_:.4f}",
        )
        ax.set_xscale("log")
        ax.set_xlabel(r"$\tau$  (log scale,  first-jump criterion)", fontsize=10)
        ax.set_ylabel("Number of clusters", fontsize=10)
        ax.set_title(
            r"Cluster-count landscape  $r_i = h_{i+1} / h_i$"
            "\n"
            r"Blue step: first-jump criterion.  "
            r"Red dashed: canonical cut cluster count (max rel-jump, independent of $\tau$)",
            fontsize=11, fontweight="bold",
        )
        ax.legend(fontsize=9)
        ax.grid(True, linewidth=0.3, alpha=0.4)

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            print(f"Saved: {save_path}")
        return fig, ax

    def plot_pipeline(
        self,
        taus: list[float] | None = None,
        n_taus: int = 6,
        save_prefix: str = "tda_graph",
        show: bool = True,
    ) -> tuple:
        """
        Run and save all three pipeline visualisations.

        Produces:
          ``{save_prefix}_01_clustering.png``
          ``{save_prefix}_02_covering.png``
          ``{save_prefix}_03_graph.png``

        Parameters
        ----------
        taus        : list of float, optional
        n_taus      : int
        save_prefix : str
        show        : bool – call plt.show() at the end

        Returns
        -------
        (fig1, axes1), (fig2, ax2), (fig3, ax3)
        """
        self._check_fitted()
        r1 = self.plot_clustering(
            taus=taus, n_taus=n_taus,
            save_path=f"{save_prefix}_01_clustering.png",
        )
        r2 = self.plot_covering(
            save_path=f"{save_prefix}_02_covering.png",
        )
        r3 = self.plot_graph(
            save_path=f"{save_prefix}_03_graph.png",
        )
        if show:
            plt.show()
        return r1, r2, r3

    # ── Colour helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _cluster_colors(labels: np.ndarray, cmap) -> list:
        u = np.unique(labels)
        denom = max(len(u) - 1, 1)
        return [cmap((lbl - 1) / denom) for lbl in labels]

    @staticmethod
    def _legend_handles(cmap) -> list:
        # Overloaded by callers that pass the current labels
        return []

    def _legend_handles(self, cmap) -> list:  # noqa: F811
        u = np.unique(self.labels_)
        denom = max(len(u) - 1, 1)
        return [
            mpatches.Patch(
                color=cmap((c - 1) / denom),
                label=f"Cluster {c}",
            )
            for c in u
        ]

    # ── Internals ─────────────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if not hasattr(self, "X_"):
            raise RuntimeError(
                "This TDAGraph instance is not fitted yet. "
                "Call fit(X) before using this method."
            )

    def __repr__(self) -> str:
        base = (
            f"TDAGraph(n_intervals={self.n_intervals}, "
            f"overlap_frac={self.overlap_frac}, "
            f"tau={self.tau!r}, "
            f"linkage_method={self.linkage_method!r})"
        )
        if not hasattr(self, "X_"):
            return base
        return (
            f"{base}\n"
            f"  fitted on {self.n_nodes_} points  |  "
            f"{self.n_clusters_} cluster(s)  |  "
            f"{self.n_patches_} patches  |  "
            f"{self.n_edges_} edges"
        )


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from sklearn.datasets import make_moons

    np.random.seed(42)
    X_demo, _ = make_moons(n_samples=150, noise=0.08)

    tda = TDAGraph(n_intervals=5, overlap_frac=0.4, tau="auto")
    tda.fit(X_demo)

    print(tda)
    print()
    tda.summary()
    print()

    # Explore the landscape
    tda.plot_tau_landscape(save_path="tda_graph_landscape.png")

    # Full pipeline
    tda.plot_pipeline(save_prefix="tda_graph_demo", show=True)
