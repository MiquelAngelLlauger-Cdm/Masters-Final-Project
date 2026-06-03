"""
mapper.diagnostics
==================

Static matplotlib diagnostics. Mirrors Sections 7 and 8 of the notebook:
  * degree distribution by tier + intra/inter edge composition,
  * cross-tier edge matrix,
  * epsilon sensitivity sweep.

All functions return the matplotlib Figure so the caller can save/show.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

from .config import TIER_LABELS, TIER_COLORS
from .graph import build_epsilon_graph


def degree_and_composition(G, params):
    """Violin of degree-by-tier + intra/inter edge bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    dd = pd.DataFrame({
        "degree": [G.degree(n) for n in G.nodes],
        "tier":   [G.nodes[n]["tier"] for n in G.nodes],
    })
    active = sorted(dd["tier"].unique())
    for i, tier in enumerate(active):
        vals = dd[dd["tier"] == tier]["degree"]
        if len(vals) == 0:
            continue
        vp = axes[0].violinplot(vals, positions=[i], showmedians=True, widths=0.6)
        for part in vp["bodies"]:
            part.set_facecolor(TIER_COLORS.get(tier, "#aaaaaa"))
            part.set_alpha(0.7)
    axes[0].set_xticks(range(len(active)))
    axes[0].set_xticklabels([f"T{t}\n{TIER_LABELS[t].split()[0]}" for t in active],
                            fontsize=8)
    axes[0].set_ylabel("Node degree")
    axes[0].set_title(f"Degree distribution by tier  (ε={params.EPSILON})",
                      fontsize=10, fontweight="bold")
    sns.despine(ax=axes[0])

    intra = sum(1 for u, v in G.edges() if G.nodes[u]["tier"] == G.nodes[v]["tier"])
    inter = sum(1 for u, v in G.edges() if G.nodes[u]["tier"] != G.nodes[v]["tier"])
    total = max(intra + inter, 1)
    axes[1].bar(["Same-tier\n(intra)", "Cross-tier\n(inter)"], [intra, inter],
                color=["#2E5FAC", "#d62728"], edgecolor="white", linewidth=0.8)
    axes[1].bar_label(axes[1].containers[0],
                      labels=[f"{intra}\n({intra/total*100:.0f}%)",
                              f"{inter}\n({inter/total*100:.0f}%)"],
                      fontsize=9, padding=3)
    axes[1].set_ylabel("N edges")
    axes[1].set_title("Edge composition", fontsize=10, fontweight="bold")
    axes[1].set_ylim(0, max(intra, inter) * 1.2 if (intra or inter) else 1)
    sns.despine(ax=axes[1])

    plt.suptitle(f"Proximity graph diagnostics  |  {params.summary()}",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    return fig


def tier_edge_matrix(G, params):
    """Heatmap of edge counts between each pair of tiers."""
    pairs = pd.DataFrame(
        [(G.nodes[u]["tier"], G.nodes[v]["tier"]) for u, v in G.edges()],
        columns=["tier_u", "tier_v"],
    )
    if not pairs.empty:
        pairs[["tier_u", "tier_v"]] = np.sort(pairs[["tier_u", "tier_v"]].values, axis=1)
    mat = pd.crosstab(pairs["tier_u"], pairs["tier_v"]) if not pairs.empty else pd.DataFrame()
    all_tiers = sorted(TIER_LABELS.keys())
    mat = mat.reindex(index=all_tiers, columns=all_tiers, fill_value=0)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(mat, annot=True, fmt="d", cmap="Blues", linewidths=0.5,
                xticklabels=[f"T{t}" for t in all_tiers],
                yticklabels=[f"T{t} {TIER_LABELS[t].split()[0]}" for t in all_tiers],
                ax=ax)
    ax.set_title(f"Edges between tier pairs  (ε={params.EPSILON})",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    return fig


def epsilon_sweep(D, df, cover, params,
                  eps_range=None):
    """
    Sweep ε and plot edge count / components / largest comp / isolated nodes.

    Uses the same edge rule as the configured pipeline so the sweep is faithful.
    """
    if eps_range is None:
        eps_range = np.arange(0.5, 8.0, 0.25)

    records = []
    for eps in eps_range:
        G_tmp = build_epsilon_graph(
            D, df, cover=cover, edge_rule=params.EDGE_RULE,
            epsilon=eps, max_bin_gap=params.MAX_BIN_GAP,
        )
        comps = list(nx.connected_components(G_tmp))
        records.append(dict(
            epsilon=eps,
            n_edges=G_tmp.number_of_edges(),
            n_components=len(comps),
            largest_comp=max((len(c) for c in comps), default=0),
            n_isolated=sum(1 for n in G_tmp.nodes if G_tmp.degree(n) == 0),
        ))
    sweep = pd.DataFrame(records)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes = axes.flatten()
    metrics = [
        ("n_edges", "N edges", "#2E5FAC"),
        ("n_components", "N components", "#d62728"),
        ("largest_comp", "Largest component", "#2ca02c"),
        ("n_isolated", "Isolated nodes", "#ff7f0e"),
    ]
    for ax, (col, label, color) in zip(axes, metrics):
        ax.plot(sweep["epsilon"], sweep[col], color=color, linewidth=2)
        ax.axvline(params.EPSILON, color="black", linestyle="--", linewidth=1.2,
                   label=f"Current ε={params.EPSILON}")
        ax.set_xlabel("ε"); ax.set_ylabel(label)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        sns.despine(ax=ax)
    plt.suptitle(f"ε sensitivity  |  edge_rule={params.EDGE_RULE}  |  "
                 f"metric={params.metric_str}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    return fig, sweep
