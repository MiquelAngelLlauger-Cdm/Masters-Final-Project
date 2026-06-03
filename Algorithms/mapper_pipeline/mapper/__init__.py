"""
mapper — modular semi-Mapper / proximity-graph pipeline for CTN-0051.

Public API
----------
>>> from mapper import MapperParams, run_pipeline, visualise
>>> params = MapperParams(LENS_KIND="density", DENSITY_METHOD="knn", DENSITY_K=10)
>>> result = run_pipeline(params)
>>> visualise(result)

Modules
-------
config      : MapperParams + tier metadata (all tunables)
data        : load/join CSVs, feature selection, standardisation
distance    : pairwise distance matrix
lenses      : feature / centrality / density projections
cover       : tunable binning (uniform or explicit edges) + memberships
graph       : ε-graph with selectable edge rules (cover / gap / none)
layout      : spring / spectral / pca node positions
viz         : Bokeh interactive plot (colour by tier or lens)
diagnostics : matplotlib degree/edge/tier/ε-sweep plots
pipeline    : end-to-end orchestration
"""

from .config import MapperParams, TIER_LABELS, TIER_COLORS
from .pipeline import run_pipeline, visualise, MapperResult

__all__ = [
    "MapperParams", "TIER_LABELS", "TIER_COLORS",
    "run_pipeline", "visualise", "MapperResult",
]
