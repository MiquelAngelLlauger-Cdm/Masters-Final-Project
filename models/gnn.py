"""
models/gnn.py
=============
Two-layer Is it  for binary retention classification on the Mapper graph.

Labels: 0 = did not complete (tiers 1-3), 1 = completed (tier 4).
Node features: lens_value, bin_idx, primary_set, in_overlap  (4 dims)
Edge weights : cosine distance stored in the 'weight' attribute.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report
import networkx as nx


NUM_CLASSES  = 2
TIER_NAMES   = ["Not completed (tiers 1-3)", "Completed (tier 4)"]


# --------------------------------------------------------------------------- #
# Graph → PyG Data
# --------------------------------------------------------------------------- #

def _build_tensors(G: nx.Graph):
    """Return raw graph tensors and label array (no masks)."""
    nodes  = sorted(G.nodes(), key=int)
    id_map = {n: i for i, n in enumerate(nodes)}

    feat_rows, labels = [], []
    for n in nodes:
        d = G.nodes[n]
        feat_rows.append([
            float(d["lens_value"]),
            float(d["bin_idx"])     / 14.0,
            float(d["primary_set"]) / 14.0,
            float(d["in_overlap"]),
        ])
        labels.append(0 if int(d["tier"]) < 4 else 1)  # tiers 1-3 → 0, tier 4 → 1

    x = torch.tensor(feat_rows, dtype=torch.float)
    y = torch.tensor(labels,    dtype=torch.long)

    src, dst, weights = [], [], []
    for u, v, ed in G.edges(data=True):
        src += [id_map[u], id_map[v]]
        dst += [id_map[v], id_map[u]]
        w    = float(ed.get("weight", 1.0))
        weights += [w, w]

    edge_index  = torch.tensor([src, dst],  dtype=torch.long)
    edge_weight = torch.tensor(weights,     dtype=torch.float)
    return x, y, edge_index, edge_weight, np.array(labels)


def _make_masks(labels, tr_idx, va_idx, te_idx):
    n = len(labels)
    def _m(idx):
        m = torch.zeros(n, dtype=torch.bool)
        m[idx] = True
        return m
    return _m(tr_idx), _m(va_idx), _m(te_idx)


def build_data(G: nx.Graph, device: str = "cpu") -> Data:
    """Convert a NetworkX Mapper graph to a PyG Data object.

    Splits nodes into train / val / test (64 / 16 / 20 %) stratified by label.
    """
    x, y, edge_index, edge_weight, labels = _build_tensors(G)

    idx = np.arange(len(labels))
    tr_idx, te_idx = train_test_split(idx, test_size=0.20, stratify=labels, random_state=42)
    tr_idx, va_idx = train_test_split(tr_idx, test_size=0.20,
                                      stratify=labels[tr_idx], random_state=42)

    data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_weight)
    data.train_mask, data.val_mask, data.test_mask = _make_masks(labels, tr_idx, va_idx, te_idx)
    return data.to(device)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

class GCN(nn.Module):
    """Two-layer GCN with a linear classification head."""

    def __init__(self, in_channels: int = 4, hidden: int = 64,
                 out_channels: int = NUM_CLASSES, dropout: float = 0.5):
        super().__init__()
        self.conv1   = GCNConv(in_channels, hidden)
        self.conv2   = GCNConv(hidden, hidden)
        self.head    = nn.Linear(hidden, out_channels)
        self.dropout = dropout

    def forward(self, data: Data) -> torch.Tensor:
        x, ei, ew = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, ei, ew))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, ei, ew))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.head(x)


# --------------------------------------------------------------------------- #
# Training helpers
# --------------------------------------------------------------------------- #

def _class_weights(data: Data, device: str) -> torch.Tensor:
    counts = torch.bincount(data.y[data.train_mask], minlength=NUM_CLASSES).float()
    w = counts.sum() / (NUM_CLASSES * counts.clamp(min=1))
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
    pred = model(data).argmax(dim=1)
    acc  = (pred[mask] == data.y[mask]).float().mean().item()
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
        patience: int = 40, device: str = "cpu") -> tuple[GCN, Data]:
    """Train and evaluate a GCN on G. Returns (model, data)."""
    data   = build_data(G, device=device)
    cw     = _class_weights(data, device)
    model  = GCN(in_channels=data.x.shape[1], hidden=hidden,
                 out_channels=NUM_CLASSES, dropout=dropout).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    n_tr, n_va, n_te = data.train_mask.sum(), data.val_mask.sum(), data.test_mask.sum()
    print(f"GCN  |  nodes={data.num_nodes}  "
          f"train={n_tr}  val={n_va}  test={n_te}")

    best_val, stale, best_state = 0.0, 0, None
    for epoch in range(1, epochs + 1):
        tr_loss        = train_step(model, data, opt, cw)
        vl_loss        = val_loss(model, data)
        val_acc, _     = evaluate(model, data, data.val_mask)

        if val_acc > best_val:
            best_val   = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            stale      = 0
        else:
            stale += 1
        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:>3d}  train_loss={tr_loss:.4f}  val_loss={vl_loss:.4f}  val_acc={val_acc:.3f}")
        if stale >= patience:
            print(f"  Early stop at epoch {epoch}")
            break

    if best_state:
        model.load_state_dict(best_state)

    test_acc, _ = evaluate(model, data, data.test_mask)
    print(f"\nBest val acc : {best_val:.3f}")
    print(f"Test acc     : {test_acc:.3f}\n")
    classification_summary(model, data, data.test_mask)
    return model, data


# --------------------------------------------------------------------------- #
# K-fold cross-validation
# --------------------------------------------------------------------------- #

def cross_validate(G: nx.Graph, k: int = 5, epochs: int = 300, lr: float = 0.01,
                   hidden: int = 64, dropout: float = 0.5,
                   patience: int = 40, device: str = "cpu",
                   random_state: int = 42) -> list[dict]:
    """Stratified k-fold CV. Returns a list of per-fold result dicts."""
    x, y, edge_index, edge_weight, labels = _build_tensors(G)

    skf     = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    idx     = np.arange(len(labels))
    results = []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(idx, labels), 1):
        tr_idx, va_idx = train_test_split(tr_idx, test_size=0.20,
                                          stratify=labels[tr_idx],
                                          random_state=random_state)

        data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_weight)
        data.train_mask, data.val_mask, data.test_mask = _make_masks(labels, tr_idx, va_idx, te_idx)
        data = data.to(device)

        cw    = _class_weights(data, device)
        model = GCN(in_channels=x.shape[1], hidden=hidden,
                    out_channels=NUM_CLASSES, dropout=dropout).to(device)
        opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

        print(f"\n── Fold {fold}/{k}  "
              f"train={data.train_mask.sum()}  val={data.val_mask.sum()}  test={data.test_mask.sum()}")

        best_val, stale, best_state = 0.0, 0, None
        for epoch in range(1, epochs + 1):
            tr_loss    = train_step(model, data, opt, cw)
            vl_loss_v  = val_loss(model, data)
            val_acc, _ = evaluate(model, data, data.val_mask)

            if val_acc > best_val:
                best_val   = val_acc
                best_state = {key: v.clone() for key, v in model.state_dict().items()}
                stale      = 0
            else:
                stale += 1
            if epoch % 5 == 0 or epoch == 1:
                print(f"  epoch {epoch:>3d}  train_loss={tr_loss:.4f}  val_loss={vl_loss_v:.4f}  val_acc={val_acc:.3f}")

            if stale >= patience:
                print(f"  Early stop at epoch {epoch}")
                break

        if best_state:
            model.load_state_dict(best_state)

        test_acc, _ = evaluate(model, data, data.test_mask)
        print(f"  → test_acc={test_acc:.3f}")
        classification_summary(model, data, data.test_mask)
        results.append({"fold": fold, "val_acc": best_val, "test_acc": test_acc})

    # ── summary table ──
    val_accs  = [r["val_acc"]  for r in results]
    test_accs = [r["test_acc"] for r in results]
    print("\n" + "─" * 42)
    print(f"{'Fold':>6}  {'Val acc':>8}  {'Test acc':>9}")
    for r in results:
        print(f"  {r['fold']:>4}  {r['val_acc']:>8.3f}  {r['test_acc']:>9.3f}")
    print(f"  Mean  {np.mean(val_accs):>8.3f}  {np.mean(test_accs):>9.3f}")
    print(f"   Std  {np.std(val_accs):>8.3f}  {np.std(test_accs):>9.3f}")
    return results
