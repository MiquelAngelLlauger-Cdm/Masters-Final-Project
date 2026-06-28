from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, f1_score
import networkx as nx


NUM_CLASSES  = 2
TIER_NAMES   = ["Not completed (tiers 1-3)", "Completed (tier 4)"]


_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "clean-data"

# Convenience presets to compose feature_cols from (paste into the notebook).
BASELINE_FEATURES = ["detox_los", "dsm_opioid_score", "sow_total_baseline"]
W8_FEATURES       = ["had_relapse_by_w8", "engagement_onset_w8"]
W12_FEATURES      = ["had_relapse_by_w12", "engagement_onset_w12"]

# Default = the w8-graph set (lens is attendance_density_w8, added automatically).
DEFAULT_FEATURE_COLS = W8_FEATURES + BASELINE_FEATURES
N_NODE_FEATURES      = 1 + len(DEFAULT_FEATURE_COLS)   # +1 for lens_value


def _patient_table() -> pd.DataFrame:
    frame = None
    for name in ("pre-trial.csv", "features_w8.csv", "features_w12.csv", "features_w24.csv"):
        path = _DATA_DIR / name
        if not path.exists():
            continue
        df    = pd.read_csv(path)
        frame = df if frame is None else frame.merge(df, on="PATID", how="outer")
    return frame.set_index("PATID")


def _build_tensors(G: nx.Graph, feature_cols: list[str] | None = None,
                   include_lens: bool = True):
    """Return raw (un-standardised) graph tensors and label array (no masks).

    ``include_lens`` adds the Mapper ``lens_value`` as node-feature 0. Set it
    False for graphs whose lens carries no class signal (e.g. the KDE-density
    lens) so the network sees only the joined ``feature_cols`` carriers.
    """
    feature_cols = list(DEFAULT_FEATURE_COLS if feature_cols is None else feature_cols)
    if not include_lens and not feature_cols:
        raise ValueError("No node features: include_lens=False with empty feature_cols.")

    nodes  = sorted(G.nodes(), key=int)
    id_map = {n: i for i, n in enumerate(nodes)}

    patids = [int(G.nodes[n]["patid"]) for n in nodes]
    labels = [0 if int(G.nodes[n]["tier"]) < 4 else 1 for n in nodes]

    # join patient features by patid, then median-impute any gaps
    table   = _patient_table()
    missing = [c for c in feature_cols if c not in table.columns]
    if missing:
        raise KeyError(f"feature_cols not found in patient CSVs: {missing}")
    feats = table.reindex(patids)[feature_cols]
    feats = feats.fillna(feats.median(numeric_only=True))

    cols = []
    if include_lens:
        cols.append(np.array([float(G.nodes[n]["lens_value"])
                              for n in nodes]).reshape(-1, 1))
    cols.append(feats.to_numpy(dtype=float))

    x = torch.tensor(np.hstack(cols), dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)

    src, dst, weights = [], [], []
    for u, v, ed in G.edges(data=True):
        src += [id_map[u], id_map[v]]
        dst += [id_map[v], id_map[u]]
        dist = float(ed.get("weight", 0.0))   # edge "weight" stores the distance
        w    = 1.0 / (1.0 + dist)             # bounded inverse distance ∈ (0, 1]
        weights += [w, w]

    edge_index  = torch.tensor([src, dst],  dtype=torch.long)
    edge_weight = torch.tensor(weights,     dtype=torch.float)
    return x, y, edge_index, edge_weight, np.array(labels)


def _set_seed(seed: int) -> None:
    """Seed NumPy and PyTorch so weight init / dropout are reproducible."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def _standardize(x: torch.Tensor, train_mask: torch.Tensor) -> torch.Tensor:
    """Z-score features using train-split statistics only (no leakage)."""
    mu = x[train_mask].mean(dim=0, keepdim=True)
    sd = x[train_mask].std(dim=0, unbiased=False, keepdim=True).clamp(min=1e-6)
    return (x - mu) / sd


def _make_masks(labels, tr_idx, va_idx, te_idx):
    n = len(labels)
    def _m(idx):
        m = torch.zeros(n, dtype=torch.bool)
        m[idx] = True
        return m
    return _m(tr_idx), _m(va_idx), _m(te_idx)


def build_data(G: nx.Graph, device: str = "cpu",
               feature_cols: list[str] | None = None,
               include_lens: bool = True) -> Data:
    x, y, edge_index, edge_weight, labels = _build_tensors(G, feature_cols, include_lens)

    idx = np.arange(len(labels))
    tr_idx, te_idx = train_test_split(idx, test_size=0.20, stratify=labels, random_state=42)
    tr_idx, va_idx = train_test_split(tr_idx, test_size=0.20,
                                      stratify=labels[tr_idx], random_state=42)

    tr_mask, va_mask, te_mask = _make_masks(labels, tr_idx, va_idx, te_idx)
    x = _standardize(x, tr_mask)

    data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_weight)
    data.train_mask, data.val_mask, data.test_mask = tr_mask, va_mask, te_mask
    return data.to(device)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

class GCN(nn.Module):
    """Two-layer GCN with a linear classification head."""

    def __init__(self, in_channels: int = N_NODE_FEATURES, hidden: int = 64,
                 out_channels: int = NUM_CLASSES, dropout: float = 0.5):
        super().__init__()
        self.conv1   = GCNConv(in_channels, hidden)
        self.conv2   = GCNConv(hidden, hidden)
        self.head    = nn.Linear(hidden, out_channels)
        self.dropout = dropout

    def forward(self, data: Data) -> torch.Tensor:
        x, ei = data.x, data.edge_index
        ew    = getattr(data, "edge_attr", None)   # bounded inverse distance
        x = F.relu(self.conv1(x, ei, edge_weight=ew))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, ei, edge_weight=ew))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.head(x)


# --------------------------------------------------------------------------- #
# Training helpers
# --------------------------------------------------------------------------- #

def _focal_loss(logits: torch.Tensor, targets: torch.Tensor,
                gamma: float = 2.0, weight: torch.Tensor | None = None) -> torch.Tensor:
    ce  = F.cross_entropy(logits, targets, weight=weight, reduction="none")
    pt  = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


def _class_weights(data: Data, device: str, minority_scale: float = 1.0) -> torch.Tensor:
    counts = torch.bincount(data.y[data.train_mask], minlength=NUM_CLASSES).float()
    w = counts.sum() / (NUM_CLASSES * counts.clamp(min=1))
    w[1] = w[1] * minority_scale
    return w.to(device)


def train_step(model: GCN, data: Data, optimizer: torch.optim.Optimizer,
               class_weights: torch.Tensor | None = None) -> float:
    model.train()
    optimizer.zero_grad()
    out  = model(data)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask],
                           weight=class_weights)
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def val_loss(model: GCN, data: Data) -> float:
    model.eval()
    out  = model(data)
    loss = F.cross_entropy(out[data.val_mask], data.y[data.val_mask])
    return loss.item()


@torch.no_grad()
def evaluate(model: GCN, data: Data, mask: torch.Tensor) -> tuple[float, torch.Tensor]:
    """Return (accuracy, predictions) on the given mask."""
    model.eval()
    pred = model(data)[mask].argmax(dim=1)
    acc  = (pred == data.y[mask]).float().mean().item()
    return acc, pred


@torch.no_grad()
def classification_summary(model: GCN, data: Data, mask: torch.Tensor) -> None:
    model.eval()
    pred = model(data)[mask].argmax(dim=1).cpu().numpy()
    true = data.y[mask].cpu().numpy()
    print(classification_report(true, pred, target_names=TIER_NAMES, zero_division=0))


# --------------------------------------------------------------------------- #
# One-call pipeline
# --------------------------------------------------------------------------- #

def run(G: nx.Graph, epochs: int = 300, lr: float = 0.01,
        hidden: int = 64, dropout: float = 0.5,
        minority_scale: float = 1.0, device: str = "cpu",
        feature_cols: list[str] | None = None,
        include_lens: bool = True, seed: int = 42) -> tuple[GCN, Data]:
    """Train and evaluate a GCN on G. Returns (model, data)."""
    _set_seed(seed)
    data   = build_data(G, device=device, feature_cols=feature_cols,
                        include_lens=include_lens)
    cw     = _class_weights(data, device, minority_scale)
    model  = GCN(in_channels=data.x.shape[1], hidden=hidden,
                 out_channels=NUM_CLASSES, dropout=dropout).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    n_tr, n_va, n_te = data.train_mask.sum(), data.val_mask.sum(), data.test_mask.sum()
    print(f"GCN  |  nodes={data.num_nodes}  train={n_tr}  val={n_va}  test={n_te}")

    for epoch in range(1, epochs + 1):

        tr_loss    = train_step(model, data, opt, cw)
        vl_loss    = val_loss(model, data)
        val_acc, _ = evaluate(model, data, data.val_mask)
        if epoch % 50 == 0 or epoch == 1:
            print(f"  epoch {epoch:>3d}  train_loss={tr_loss:.4f}  val_loss={vl_loss:.4f}  val_acc={val_acc:.3f}")

    test_acc, _ = evaluate(model, data, data.test_mask)
    print(f"\nTest acc: {test_acc:.3f}\n")
    classification_summary(model, data, data.test_mask)
    return model, data


# --------------------------------------------------------------------------- #
# K-fold cross-validation
# --------------------------------------------------------------------------- #

def cross_validate(G: nx.Graph, k: int = 5, epochs: int = 300, lr: float = 0.01,
                   hidden: int = 64, dropout: float = 0.5,
                   minority_scale: float = 1.0, focal_gamma: float | None = None,
                   device: str = "cpu", random_state: int = 42,
                   feature_cols: list[str] | None = None,
                   include_lens: bool = True) -> list[dict]:
    """Stratified k-fold CV (train/test split only). Returns a list of per-fold result dicts."""
    _set_seed(random_state)
    x, y, edge_index, edge_weight, labels = _build_tensors(G, feature_cols, include_lens)

    skf     = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    idx     = np.arange(len(labels))
    results = []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(idx, labels), 1):
        n = len(labels)
        def _m(i):
            m = torch.zeros(n, dtype=torch.bool); m[i] = True; return m

        tr_mask, te_mask = _m(tr_idx), _m(te_idx)
        data = Data(x=_standardize(x, tr_mask), y=y,
                    edge_index=edge_index, edge_attr=edge_weight)
        data.train_mask = tr_mask
        data.test_mask  = te_mask
        data = data.to(device)

        cw    = _class_weights(data, device, minority_scale)
        model = GCN(in_channels=x.shape[1], hidden=hidden,
                    out_channels=NUM_CLASSES, dropout=dropout).to(device)
        opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

        print(f"\n── Fold {fold}/{k}  train={data.train_mask.sum()}  test={data.test_mask.sum()}")

        for epoch in range(1, epochs + 1):
            model.train()
            opt.zero_grad()
            out  = model(data)
            if focal_gamma is not None:
                loss = _focal_loss(out[data.train_mask], data.y[data.train_mask], gamma=focal_gamma, weight=cw)
            else:
                loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask], weight=cw)
            loss.backward()
            opt.step()
            if epoch % 20 == 0 or epoch == 1:
                print(f"  epoch {epoch:>3d}  train_loss={loss.item():.4f}")

        test_acc, preds = evaluate(model, data, data.test_mask)
        true = data.y[data.test_mask].cpu().numpy()
        test_f1 = f1_score(true, preds.cpu().numpy(), average="macro", zero_division=0)
        print(f"  → test_acc={test_acc:.3f}  test_f1={test_f1:.3f}")
        classification_summary(model, data, data.test_mask)
        results.append({"fold": fold, "test_acc": test_acc, "test_f1": test_f1})

    # ── summary table ──
    test_accs = [r["test_acc"] for r in results]
    test_f1s  = [r["test_f1"]  for r in results]
    print("\n" + "─" * 40)
    print(f"{'Fold':>6}  {'Test acc':>9}  {'Test F1':>9}")
    for r in results:
        print(f"  {r['fold']:>4}  {r['test_acc']:>9.3f}  {r['test_f1']:>9.3f}")
    print(f"  Mean  {np.mean(test_accs):>9.3f}  {np.mean(test_f1s):>9.3f}")
    print(f"   Std  {np.std(test_accs):>9.3f}  {np.std(test_f1s):>9.3f}")
    return results
