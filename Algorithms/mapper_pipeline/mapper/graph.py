"""
mapper.graph
============

ε-neighbourhood graph construction with selectable edge-pruning rules.

The original notebook built the graph three different ways in three cells:
  * tier-gap pruning,
  * bin-gap pruning,
  * cover-set sharing (the actual Mapper rule).

Here they are unified behind a single ``edge_rule`` argument so the choice is a
parameter, not a copy-pasted cell.

Edge rules (all also require proximity ``d(i,j) <= EPSILON``):
  * ``"cover"`` : keep edge iff i and j share >= 1 cover set (Mapper condition).
  * ``"gap"``   : keep edge iff |bin_index[i] - bin_index[j]| <= MAX_BIN_GAP.
  * ``None``    : keep all proximity edges (used to build the lens-input graph
                  for the centrality lens).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import networkx as nx

from .config import TIER_LABELS


def build_epsilon_graph(
    D: np.ndarray,
    df,
    cover=None,
    lens=None,
    edge_rule: Optional[str] = "cover",
    epsilon: float = 5.0,
    max_bin_gap: int = 1,
) -> nx.Graph:
    """
    Build the ε-neighbourhood graph with the requested edge rule.

    Parameters
    ----------
    D : (n, n) distance matrix
    df : patient dataframe (for node attributes)
    cover : Cover or None
        Required for edge_rule in {"cover", "gap"}.
    lens : LensResult or None
        Attached to nodes for hover/colour; optional.
    edge_rule : {"cover", "gap", None}
    epsilon : float
    max_bin_gap : int
        Used only when edge_rule == "gap".

    Returns
    -------
    networkx.Graph with integer node ids 0..n-1.
    """
    n = len(df)
    tiers = df["retention_tier"].to_numpy()
    G = nx.Graph()

    # ---- nodes ---------------------------------------------------------- #
    memberships = cover.memberships if cover is not None else [[] for _ in range(n)]
    bin_index   = cover.bin_index   if cover is not None else np.full(n, -1)

    for i in range(n):
        t = int(tiers[i]) if not np.isnan(tiers[i]) else -1
        attrs = dict(
            patid       = int(df["PATID"].iloc[i]),
            tier        = t,
            tier_label  = TIER_LABELS.get(t, "Unknown"),
            memberships = memberships[i],
            primary_set = memberships[i][0] if memberships[i] else -1,
            bin_idx     = int(bin_index[i]),
            in_overlap  = len(memberships[i]) > 1,
        )
        if lens is not None:
            attrs["lens_value"] = float(lens.values[i])
        G.add_node(i, **attrs)

    # ---- edges ---------------------------------------------------------- #
    kept = pruned = skipped = 0
    for i in range(n):
        for j in range(i + 1, n):
            if D[i, j] > epsilon:
                continue
            keep, reason = _edge_keep(
                i, j, edge_rule, memberships, bin_index, max_bin_gap
            )
            if keep:
                data = dict(weight=float(D[i, j]))
                if edge_rule == "cover":
                    shared = sorted(set(memberships[i]) & set(memberships[j]))
                    data["shared_sets"] = shared
                elif edge_rule == "gap":
                    data["bin_gap"] = abs(int(bin_index[i]) - int(bin_index[j]))
                G.add_edge(i, j, **data)
                kept += 1
            elif reason == "skip":
                skipped += 1
            else:
                pruned += 1

    G.graph.update(edges_kept=kept, edges_pruned=pruned, edges_skipped=skipped,
                   epsilon=epsilon, edge_rule=str(edge_rule))
    return G


def _edge_keep(i, j, edge_rule, memberships, bin_index, max_bin_gap):
    """Return (keep: bool, reason: str)."""
    if edge_rule is None:
        return True, "proximity"

    if edge_rule == "cover":
        mi, mj = set(memberships[i]), set(memberships[j])
        if not mi or not mj:
            return False, "skip"          # missing lens value -> no set
        return (bool(mi & mj), "cover")

    if edge_rule == "gap":
        bi, bj = int(bin_index[i]), int(bin_index[j])
        if bi < 0 or bj < 0:
            return False, "skip"
        return (abs(bi - bj) <= max_bin_gap, "gap")

    raise ValueError(f"Unknown edge_rule '{edge_rule}'.")


def describe_graph(G: nx.Graph) -> str:
    """Summary string mirroring the notebook prints."""
    n_iso = sum(1 for nd in G.nodes if G.degree(nd) == 0)
    lines = [
        f"Nodes            : {G.number_of_nodes()}",
        f"Edges (kept)     : {G.graph.get('edges_kept', G.number_of_edges())}",
        f"Edges (pruned)   : {G.graph.get('edges_pruned', 0)}",
        f"Edges (skipped)  : {G.graph.get('edges_skipped', 0)}",
        f"Connected comps  : {nx.number_connected_components(G)}",
        f"Isolated nodes   : {n_iso}",
    ]
    # mean degree by tier
    lines.append("\nMean degree by tier:")
    for tier, label in TIER_LABELS.items():
        nodes_t = [nd for nd in G.nodes if G.nodes[nd]["tier"] == tier]
        if nodes_t:
            md = np.mean([G.degree(nd) for nd in nodes_t])
            lines.append(f"  Tier {tier} ({label:<14s}): "
                         f"{len(nodes_t):3d} nodes, mean deg = {md:.1f}")
    return "\n".join(lines)
