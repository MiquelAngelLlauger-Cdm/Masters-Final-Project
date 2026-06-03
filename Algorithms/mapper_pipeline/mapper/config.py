"""
mapper.config
=============

All tunable parameters in one place. This is the modular replacement for the
notebook's "Section 2 — tune here" cell.

Edit a ``MapperParams`` instance (or build one in your driver script/notebook)
and pass it through the pipeline. Nothing else needs editing to change the
analysis.

The binning of the lens stays fully tunable here:
  * For a *uniform Mapper cover* use ``N_INTERVALS`` + ``OVERLAP`` (resolution /
    gain), which is the standard Mapper parameterisation.
  * For *explicit / irregular bins* (the notebook's ``BIN_EDGES`` style) set
    ``COVER_MODE = "edges"`` and provide ``BIN_EDGES`` (+ optional ``BIN_LABELS``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class MapperParams:
    # ----------------------------------------------------------------- #
    # Data paths
    # ----------------------------------------------------------------- #
    PRE_TRIAL_CSV: str = "../data/clean-data/pre-trial.csv"
    TARGET_CSV:    str = "../data/clean-data/retention_tier.csv"
    W8_CSV:        str = "../data/clean-data/features_w8.csv"
    W12_CSV:       str = "../data/clean-data/features_w12.csv"
    JOIN_KEY:      str = "PATID"

    # ----------------------------------------------------------------- #
    # Feature selection (mirrors notebook Section 1)
    # ----------------------------------------------------------------- #
    # Columns to drop outright.
    EXCLUDE_COLS: List[str] = field(default_factory=lambda: [
        "PATID", "retention_tier", "retention_tier_label",
        "last_opioid_date",
        "last_substance_label", "last_route_label", "discharge_facility_label",
        "detox_admission_day", "detox_discharge_day",
        "opi_past_12_months",
        "opi_withdrawal", "opi_tolerance", "opi_strong_craving",
        "opi_social_problems", "opi_continued_use_despite_problems",
        "opi_larger_longer_dose", "opi_cut_back_unsuccess",
        "opi_time_spent", "opi_activities_reduced",
        "opi_recurrent_obligat_fail", "opi_hazardous_use",
        # 'attendance_density_w8', "had_relapse_by_w8", "engagement_onset_w8",
        'attendance_density_w12', "had_relapse_by_w12", "engagement_onset_w12"
    ])
    # Prefixes / suffixes to drop (SOW, DSM composites, merge artifacts).
    EXCLUDE_PREFIXES: List[str] = field(default_factory=lambda: ["sow_", "dsm_"])
    EXCLUDE_SUFFIXES: List[str] = field(default_factory=lambda: ["_baseline_x",
                                                                 "_baseline_y"])
    STANDARDISE: bool = True   # StandardScaler on feature matrix

    # ----------------------------------------------------------------- #
    # Distance / proximity
    # ----------------------------------------------------------------- #
    EPSILON: float = 5.0           # neighbourhood radius (standardised space)
    METRIC: str = "euclidean"      # euclidean | cosine | manhattan | minkowski
    MINKOWSKI_P: int = 3           # only used when METRIC == "minkowski"

    # ----------------------------------------------------------------- #
    # Lens / projection  (the new modular extension)
    # ----------------------------------------------------------------- #
    LENS_KIND: str = "feature"     # "feature" | "centrality" | "density"

    # feature lens
    FEATURE_LENS_COL: str = "attendance_density_w8"

    # centrality lens
    CENTRALITY_MEASURE: str = "degree"   # degree | betweenness | closeness |
                                         # eigenvector | pagerank | eccentricity

    # density lens
    DENSITY_METHOD: str = "knn"          # knn | knn_dist | ball | kde
    DENSITY_K: int = 10
    DENSITY_BANDWIDTH: Optional[float] = None  # None -> median-distance default

    # ----------------------------------------------------------------- #
    # Cover / binning  (kept fully tunable)
    # ----------------------------------------------------------------- #
    COVER_MODE: str = "uniform"    # "uniform" | "edges"

    # uniform cover (standard Mapper resolution/gain)
    N_INTERVALS: int = 26          # number of overlapping intervals
    OVERLAP: float = 0.5           # fractional overlap between adjacent sets
    COVER_RANGE: Optional[Tuple[float, float]] = None  # None -> [min, max] of lens

    # explicit-edge cover (notebook BIN_EDGES style)
    BIN_EDGES: Optional[List[float]] = None
    BIN_LABELS: Optional[List[str]] = None
    BIN_RIGHT: bool = False        # pandas.cut right-closed? (notebook used False)

    # piecewise cover: list of (lo, hi, n_intervals) segments, with OVERLAP applied
    # within each segment. Segments should partition COVER_RANGE (or lens range).
    PIECEWISE_SEGMENTS: list[tuple[float, float, int]] | None = None
    # Example: [(0.0, 0.5, 40), (0.5, 1.0, 15)]  -> 40 intervals dense, 15 sparse


    # ----------------------------------------------------------------- #
    # Edge rule for the *displayed* Mapper graph
    # ----------------------------------------------------------------- #
    # "cover" : connect proximate patients sharing >=1 cover set (Mapper rule)
    # "gap"   : connect proximate patients whose cover-bin index differs by
    #           <= MAX_BIN_GAP (the notebook's tier/bin-gap pruning)
    # "none"  : pure proximity graph (no lens pruning)
    EDGE_RULE: str = "cover"
    MAX_BIN_GAP: int = 1

    # ----------------------------------------------------------------- #
    # Layout & rendering
    # ----------------------------------------------------------------- #
    LAYOUT: str = "spring"         # spring | spectral | pca
    SPRING_K: float = 5.0
    SPRING_ITERS: int = 50
    SEED: int = 42

    # node colouring in the Bokeh plot: "tier" | "lens"
    COLOR_BY: str = "tier"

    FIG_WIDTH: int = 900
    FIG_HEIGHT: int = 700

    # ----------------------------------------------------------------- #
    # Derived helpers
    # ----------------------------------------------------------------- #
    @property
    def metric_str(self) -> str:
        if self.METRIC == "minkowski":
            return f"minkowski(p={self.MINKOWSKI_P})"
        return self.METRIC

    @property
    def dist_kwargs(self) -> dict:
        return {"p": self.MINKOWSKI_P} if self.METRIC == "minkowski" else {}

    def summary(self) -> str:
        lens = self.LENS_KIND
        if lens == "feature":
            lens += f"={self.FEATURE_LENS_COL}"
        elif lens == "centrality":
            lens += f"={self.CENTRALITY_MEASURE}"
        elif lens == "density":
            lens += f"={self.DENSITY_METHOD}"
        cover = (f"uniform({self.N_INTERVALS}x{self.OVERLAP:.0%})"
                 if self.COVER_MODE == "uniform" else "edges")
        return (f"ε={self.EPSILON} | metric={self.metric_str} | lens={lens} | "
                f"cover={cover} | edge_rule={self.EDGE_RULE} | layout={self.LAYOUT}")


# Tier metadata used throughout (kept here so it is configurable in one place).
TIER_LABELS = {
    0: "Never engaged",
    1: "Early dropout",
    2: "Mid dropout",
    3: "Late dropout",
    4: "Completed",
}

TIER_COLORS = {
    -1: "#aaaaaa",
     0: "#d62728",
     1: "#cd6001",
     2: "#f99430",
     3: "#92f67e",
     4: "#028002",
}
