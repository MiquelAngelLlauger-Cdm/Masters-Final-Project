# Modular semi-Mapper pipeline (CTN-0051)

A refactor of the original `proximity_graph.ipynb` notebook into a clean,
tunable Python package. It builds an ε-neighbourhood proximity graph on the
baseline feature matrix, prunes edges with a Mapper-style cover rule, and
renders it interactively with Bokeh — now with **three interchangeable lenses**.

## What changed vs. the notebook

The notebook did everything inline and rebuilt the graph three times in three
cells (tier-gap, bin-gap, cover-sharing). Here each concern is one module, and
the three graph-building variants are unified behind a single `EDGE_RULE`
parameter. The big addition you asked for: the **projection / filter function
("lens")** is now pluggable.

| Lens | What it projects onto | Config |
|------|----------------------|--------|
| `feature`    | a raw data column (original behaviour) | `FEATURE_LENS_COL` |
| `centrality` | graph centrality of the proximity graph | `CENTRALITY_MEASURE` = degree / betweenness / closeness / eigenvector / pagerank / eccentricity |
| `density`    | local density in feature space | `DENSITY_METHOD` = knn / knn_dist / ball / kde, `DENSITY_K`, `DENSITY_BANDWIDTH` |

Everything else you wanted preserved is preserved: the **Bokeh** visualisation
(hover, tier legend, styling — plus an optional colour-by-lens mode), all
**parameters** (ε, metric, layout, etc.), and **tunable binning** via the
`cover` module.

## Layout

```
mapper_pipeline/
├── run_mapper.ipynb        # thin driver notebook — tune params & run
├── README.md
└── mapper/
    ├── __init__.py         # public API
    ├── config.py           # MapperParams (all tunables) + tier metadata
    ├── data.py             # load/join CSVs, feature selection, standardise
    ├── distance.py         # pairwise distance matrix
    ├── lenses.py           # feature / centrality / density projections  ← new
    ├── cover.py            # tunable binning (uniform or explicit edges)
    ├── graph.py            # ε-graph with cover / gap / none edge rules
    ├── layout.py           # spring / spectral / pca positions
    ├── viz.py              # Bokeh interactive plot
    ├── diagnostics.py      # matplotlib degree/edge/tier/ε-sweep plots
    └── pipeline.py         # end-to-end orchestration
```

## Usage

```python
from mapper import MapperParams, run_pipeline, visualise

params = MapperParams(
    LENS_KIND="density", DENSITY_METHOD="knn", DENSITY_K=10,
    EPSILON=5.0, COVER_MODE="uniform", N_INTERVALS=26, OVERLAP=0.5,
    EDGE_RULE="cover", COLOR_BY="lens",
)
result = run_pipeline(params)
visualise(result)
```

### Tuning the binning

Two cover modes, both fully tunable:

* **Uniform Mapper cover** (resolution / gain):
  `COVER_MODE="uniform"`, `N_INTERVALS=26`, `OVERLAP=0.5`.
* **Explicit / clinical bins** (the notebook's `BIN_EDGES` style):
  `COVER_MODE="edges"`, `BIN_EDGES=[0,3,7,14,21]`,
  `BIN_LABELS=["0-3","3-7","7-14","14+"]`.

### Edge rules

* `EDGE_RULE="cover"` — connect proximate patients sharing ≥1 cover set (Mapper).
* `EDGE_RULE="gap"`   — connect if cover-bin indices differ by ≤ `MAX_BIN_GAP`.
* `EDGE_RULE="none"`  — pure proximity graph.

## Lens dependency note

The `centrality` lens is computed from a proximity-only graph, so the pipeline
builds that graph first, derives the lens, *then* builds the final pruned
Mapper graph. `feature` and `density` lenses don't depend on the graph. The
orchestrator handles this ordering automatically.

## Dependencies

`pandas`, `numpy`, `scikit-learn`, `networkx`, `bokeh`, `matplotlib`, `seaborn`.
