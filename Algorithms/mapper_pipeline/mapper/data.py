"""
mapper.data
===========

Data loading, joining, feature selection and standardisation.
Mirrors Sections 1 and 3 of the original notebook.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class Dataset:
    """Bundle of everything downstream stages need."""
    df: pd.DataFrame          # joined patient table
    feat_cols: List[str]      # columns used as the feature matrix
    X: np.ndarray             # (n, d) standardised (or raw) feature matrix
    scaler: object            # fitted StandardScaler or None


def load_dataset(params) -> Dataset:
    """
    Load and join the three CSVs, select feature columns, standardise.

    Returns
    -------
    Dataset
    """
    pre_trial = pd.read_csv(params.PRE_TRIAL_CSV)
    target    = pd.read_csv(params.TARGET_CSV)
    w8        = pd.read_csv(params.W8_CSV)
    w12       = pd.read_csv(params.W12_CSV)

    key = params.JOIN_KEY
    df = (
        pre_trial
        .merge(target[[key, "retention_tier"]], on=key, how="inner")
        .merge(w8, on=key, how="inner")
        .merge(w12, on=key, how="inner")
    )

    feat_cols = select_features(df, params)
    X_raw = df[feat_cols].to_numpy(dtype=float)

    if params.STANDARDISE:
        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)
    else:
        scaler = None
        X = X_raw

    return Dataset(df=df, feat_cols=feat_cols, X=X, scaler=scaler)


def select_features(df: pd.DataFrame, params) -> List[str]:
    """Apply the exclude lists / prefixes / suffixes from the notebook."""
    cols = []
    for c in df.columns:
        if c in params.EXCLUDE_COLS:
            continue
        if df[c].dtype == object:
            continue
        if any(c.startswith(p) for p in params.EXCLUDE_PREFIXES):
            continue
        if any(c.endswith(s) for s in params.EXCLUDE_SUFFIXES):
            continue
        cols.append(c)
    return cols


def describe_dataset(ds: Dataset, params) -> str:
    """Human-readable summary, equivalent to the notebook's prints."""
    df = ds.df
    lines = [
        f"Patients        : {len(df)}",
        f"Feature matrix  : {ds.X.shape}",
        f"Missing values  : {int(np.isnan(ds.X).sum())}",
        "",
        "Retention tier distribution:",
        df["retention_tier"].value_counts().sort_index().to_string(),
    ]
    return "\n".join(lines)
