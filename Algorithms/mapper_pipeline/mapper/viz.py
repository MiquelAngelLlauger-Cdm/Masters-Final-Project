"""
mapper.viz
==========

Interactive Bokeh visualisation. Mirrors Section 6 of the notebook, preserving
the node/edge styling, hover tooltips and tier legend. Adds the option to
colour nodes by the lens value (``params.COLOR_BY = "lens"``) in addition to the
original tier colouring (``"tier"``).
"""

from __future__ import annotations

import numpy as np
import networkx as nx

from bokeh.plotting import figure, show
from bokeh.models import (
    ColumnDataSource, HoverTool, Legend, LegendItem,
    LinearColorMapper, ColorBar,
)
from bokeh.palettes import Viridis256

from .config import TIER_LABELS, TIER_COLORS


def build_sources(G: nx.Graph, df, lens=None, params=None):
    """Build Bokeh ColumnDataSources for nodes and edges."""
    nodes = list(G.nodes)

    node_tier  = [G.nodes[n]["tier"] for n in nodes]
    node_lens  = [G.nodes[n].get("lens_value", float("nan")) for n in nodes]

    # ---- colour ---------------------------------------------------------- #
    color_by = (params.COLOR_BY if params is not None else "tier")
    if color_by == "lens" and lens is not None:
        vals = np.array(node_lens, dtype=float)
        finite = vals[np.isfinite(vals)]
        vmin, vmax = (float(finite.min()), float(finite.max())) if finite.size else (0, 1)
        mapper = LinearColorMapper(palette=Viridis256, low=vmin, high=vmax)
        node_color = None   # handled by transform at draw time
    else:
        mapper = None
        node_color = [TIER_COLORS.get(t, "#aaaaaa") for t in node_tier]

    def col(name, default=float("nan")):
        return [float(df[name].iloc[n]) if name in df.columns else default
                for n in nodes]

    node_source = ColumnDataSource(dict(
        x          = [G.nodes[n]["x"] for n in nodes],
        y          = [G.nodes[n]["y"] for n in nodes],
        tier       = [str(t) for t in node_tier],
        patid      = [G.nodes[n]["patid"] for n in nodes],
        degree     = [G.degree(n) for n in nodes],
        lens_value = node_lens,
        primary_set= [G.nodes[n]["primary_set"] for n in nodes],
        in_overlap = [str(G.nodes[n]["in_overlap"]) for n in nodes],
        detox_los  = col("detox_los"),
        attendance_w8 = col("attendance_density_w8"),
        size       = [8] * len(nodes),
    ))
    if node_color is not None:
        node_source.data["color"] = node_color

    # ---- edges ----------------------------------------------------------- #
    edge_xs, edge_ys, edge_w, edge_shared = [], [], [], []
    for u, v, data in G.edges(data=True):
        edge_xs.append([G.nodes[u]["x"], G.nodes[v]["x"]])
        edge_ys.append([G.nodes[u]["y"], G.nodes[v]["y"]])
        edge_w.append(data.get("weight", 0.0))
        # # shared cover sets beyond 1 (used for emphasis, like the notebook)
        edge_shared.append(len(data.get("shared_sets", [1])) - 1)

    edge_alpha = [0.8 if g == 0 else 0.25 for g in edge_shared]
    edge_color = ["#363636" if g == 0 else "#aaaaaa" for g in edge_shared]

    edge_source = ColumnDataSource(dict(
        xs=edge_xs, ys=edge_ys, weight=edge_w,
        shared=edge_shared, alpha=edge_alpha, color=edge_color,
    ))

    return node_source, edge_source, mapper


def make_figure(G, node_source, edge_source, mapper, lens, params):
    """Assemble the Bokeh figure with edges, nodes, hover and legend."""
    title = (f"Mapper proximity graph  |  {params.summary()}\n"
             f"{G.number_of_nodes()} patients · {G.number_of_edges()} edges  "
             f"({nx.number_connected_components(G)} components)")

    p = figure(
        title=title,
        tools="pan,wheel_zoom,box_zoom,reset,save",
        width=params.FIG_WIDTH, height=params.FIG_HEIGHT,
        background_fill_color="#FAFAFA",
    )
    p.title.text_font_size = "11pt"
    p.title.text_font_style = "normal"
    p.axis.visible = False
    p.grid.visible = False
    p.outline_line_color = "#DDDDDD"

    # edges
    p.multi_line(xs="xs", ys="ys", line_color="color",
                 line_alpha="alpha", line_width=1, source=edge_source)

    # nodes
    if mapper is not None:   # colour by lens
        from bokeh.transform import transform
        nodes_r = p.scatter(
            x="x", y="y", size="size",
            fill_color=transform("lens_value", mapper),
            fill_alpha=0.5, line_color="white", line_width=0.8,
            source=node_source,
        )
        cbar = ColorBar(color_mapper=mapper, width=8,
                        title=lens.label if lens else "lens")
        p.add_layout(cbar, "right")
    else:                    # colour by tier
        nodes_r = p.scatter(
            x="x", y="y", size="size",
            fill_color="color", fill_alpha=0.6,
            line_color="white", line_width=0.8,
            source=node_source,
        )
        _add_tier_legend(p, node_source)

    # hover
    p.add_tools(HoverTool(renderers=[nodes_r], tooltips=[
        ("PATID", "@patid"),
        ("Tier", "@tier"),
        ("Degree", "@degree connections"),
        ("Lens value", "@lens_value{0.000}"),
        ("Cover set", "@primary_set"),
        ("In overlap", "@in_overlap"),
        ("Detox LOS", "@detox_los{0.0} days"),
        ("Attendance (w8)", "@attendance_w8{0.000}"),
    ]))

    return p


def _add_tier_legend(p, node_source):
    node_tier = [int(t) for t in node_source.data["tier"]]
    items = []
    for tier_val, label in TIER_LABELS.items():
        n_tier = sum(1 for t in node_tier if t == tier_val)
        if n_tier == 0:
            continue
        dummy = p.scatter(x=[float("nan")], y=[float("nan")], size=10,
                          fill_color=TIER_COLORS[tier_val], line_color="white",
                          line_width=0.8, fill_alpha=0.85)
        items.append(LegendItem(label=f"Tier {tier_val}: {label} (n={n_tier})",
                                renderers=[dummy]))
    legend = Legend(items=items, location="top_right",
                    background_fill_alpha=0.9, border_line_color="#CCCCCC",
                    label_text_font_size="9pt", glyph_width=12, glyph_height=12,
                    spacing=4)
    p.add_layout(legend, "right")


def visualise(G, df, lens, params, render=True):
    """High-level entry point: build sources, figure, optionally show()."""
    node_source, edge_source, mapper = build_sources(G, df, lens, params)
    p = make_figure(G, node_source, edge_source, mapper, lens, params)
    if render:
        show(p)
    return p
