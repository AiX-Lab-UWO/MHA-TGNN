# ============================================================
# DGW — Fixed-Topology + Edge Attributes + CB-Focal (no sampler)
# ============================================================

import os
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import TransformerConv, AttentionalAggregation as GlobalAttention

# -----------------------------
# Paths (EDIT if needed)
# -----------------------------
val_dir   = r'...\DGW_Dataset\val_cleaned_landmarks.csv'
train_dir = r'...\DGW_Dataset\train_cleaned_landmarks.csv'

# -----------------------------
# Config
# -----------------------------
NUM_CLASSES   = 9
EPOCHS        = 50
BATCH_SIZE    = 64
LR            = 5e-4
WEIGHT_DECAY  = 1e-4
STEP_SIZE     = 20
GAMMA         = 0.5
SEED          = 1337

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Graph topology: fixed edges + edge attributes
SYMMETRIC_EDGES = False
USE_EDGE_ATTR   = True
EDGE_ATTR_DIM   = 4   # [ux, uy, uz, log1p(|d|)]
EPS             = 1e-6

# GNN
HEADS  = 8
LAYERS = 3

# CB-Focal (imbalance)
CB_BETA          = 0.995    # 0.99–0.999 usually
FOCAL_GAMMA      = 1.5      # 1.0–2.0
LABEL_SMOOTHING  = 0.05

# Outputs
VAL_PRED_CSV = "dgw_val_predictions.csv"
SAVED_MODEL  = "dgw_edgeattr_cbfocal_model.pth"

# -----------------------------
# Repro
# -----------------------------
def set_seed(seed=SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(SEED)

# ---------------------------------------
# Fixed baseline edges (~998), clipped by N
# ---------------------------------------
def get_fixed_edges(n_nodes: int) -> torch.Tensor:
    # pupils hub connections + eyelid/nose ring snippets
    edges = [(468, node) for node in range(n_nodes) if node != 468]
    edges += [(473, node) for node in range(n_nodes) if node != 473]
    extras = [
        (471, 159), (159, 469), (469, 145), (145, 471),
        (476, 475), (475, 474), (474, 477), (477, 476),
        (1, 33), (1, 173), (1, 162), (1, 263), (1, 398), (1, 368),
        (33, 246), (146, 161), (161, 160), (160, 150), (150, 158),
        (158, 157), (157, 173), (173, 155), (155, 154), (154, 153),
        (153, 145), (145, 144), (144, 163), (163, 7), (7, 33),
        (398, 384), (384, 385), (385, 386), (386, 387), (387, 388),
        (388, 263), (263, 249), (249, 390), (390, 373), (373, 374),
        (374, 380), (380, 381), (381, 382), (382, 398)
    ]
    edges += [(u, v) for (u, v) in extras if 0 <= u < n_nodes and 0 <= v < n_nodes]
    # unique undirected pairs
    E = set()
    for (u, v) in edges:
        if u == v: continue
        a, b = (u, v) if u < v else (v, u)
        E.add((a, b))
    rows, cols = [], []
    for (u, v) in sorted(E):
        rows.append(u); cols.append(v)
        if SYMMETRIC_EDGES:
            rows.append(v); cols.append(u)
    edge_index = np.vstack([rows, cols]).astype(np.int64)  # [2, E]
    return torch.from_numpy(edge_index)

# ---------------------------------------
# Geometry & edge attributes
# ---------------------------------------
def normalize_coords(coords_np: np.ndarray) -> np.ndarray:
    c = coords_np.astype(np.float32)
    center = c.mean(axis=0, keepdims=True)
    c0 = c - center
    rms = np.sqrt((c0**2).sum(axis=1).mean()) + EPS
    return c0 / rms

def build_edge_attr_for_fixed_edges(coords_norm: np.ndarray, edge_index_np: np.ndarray) -> torch.Tensor:
    src = coords_norm[edge_index_np[0]]   # [E,3]
    dst = coords_norm[edge_index_np[1]]   # [E,3]
    vec = dst - src
    dist = np.linalg.norm(vec, axis=1, keepdims=True)  # [E,1]
    unit = vec / np.clip(dist, EPS, None)
    logd = np.log1p(dist)
    ea = np.concatenate([unit, logd], axis=1).astype(np.float32)  # [E,4]
    return torch.from_numpy(ea)

# ---------------------------------------
# CSV parsing for DGW format
#   col[0]   = filename
#   col[1:-1]= flattened 3D landmarks (any order as long as triplets)
#   col[-1]  = Label in [1..9]
# ---------------------------------------
def parse_row_dgw(row: pd.Series):
    filename = str(row.iloc[0])
    label_raw = int(row.iloc[-1])
    label = label_raw - 1  # map 1..9 -> 0..8
    vals = row.iloc[1:-1].to_numpy(dtype=np.float32)
    L = (len(vals) // 3) * 3
    vals = vals[:L]
    coords = vals.reshape(-1, 3).astype(np.float32)
    return filename, coords, label

# ---------------------------------------
# Build PyG datasets from CSV
# ---------------------------------------
def build_graphs_from_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    data_list = []
    for _, row in df.iterrows():
        fname, coords_raw, label = parse_row_dgw(row)
        coords_norm = normalize_coords(coords_raw)
        n_nodes = coords_norm.shape[0]

        edge_index = get_fixed_edges(n_nodes)      # [2,E]
        ei_np = edge_index.numpy()
        edge_attr = build_edge_attr_for_fixed_edges(coords_norm, ei_np) if USE_EDGE_ATTR else None

        x = torch.from_numpy(coords_norm)
        y = torch.tensor([label], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index.long(), edge_attr=edge_attr, y=y)
        data.fname = fname  # keep filename
        data_list.append(data)
    return data_list

# ---------------------------------------
# ==== CB-Focal components (no sampler)
# ---------------------------------------
def get_class_counts(graphs, num_classes):
    counts = np.zeros(num_classes, dtype=np.int64)
    for d in graphs:
        counts[int(d.y.item())] += 1
    return counts

def class_balanced_weights(counts, beta=CB_BETA):
    counts = np.asarray(counts, dtype=np.float64)
    eff_num = 1.0 - np.power(beta, counts)
    eff_num[eff_num <= 0] = 1e-12
    w = (1.0 - beta) / eff_num
    return (w / (w.mean() + 1e-12)).astype(np.float32)  # average weight ≈ 1

class CBFocalLoss(nn.Module):
    def __init__(self, class_weights: torch.Tensor,
                 gamma: float = FOCAL_GAMMA,
                 label_smoothing: float = LABEL_SMOOTHING):
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits, target):
        ce = F.cross_entropy(
            logits, target,
            weight=self.class_weights,
            reduction='none',
            label_smoothing=self.label_smoothing
        )
        with torch.no_grad():
            pt = F.softmax(logits, dim=1).gather(1, target.view(-1, 1)).squeeze(1)
            pt = pt.clamp_(1e-6, 1.0 - 1e-6)
        loss = ((1.0 - pt) ** self.gamma) * ce
        return loss.mean()

# ---------------------------------------
# Model: TransformerConv + edge_attr
# ---------------------------------------
class TransformerNet(nn.Module):
    def __init__(self, num_node_features, output_dim):
        super().__init__()
        h1, h2, h3 = 64, 32, 16
        self.conv1 = TransformerConv(
            num_node_features, h1 * 8, heads=HEADS, concat=False,
            edge_dim=(EDGE_ATTR_DIM if USE_EDGE_ATTR else None),
            beta=True, dropout=0.10
        )
        self.conv2 = TransformerConv(
            h1 * 8, h2 * 8, heads=HEADS, concat=False,
            edge_dim=(EDGE_ATTR_DIM if USE_EDGE_ATTR else None),
            beta=True, dropout=0.10
        )
        self.conv3 = TransformerConv(
            h2 * 8, h3 * 4, heads=HEADS, concat=False,
            edge_dim=(EDGE_ATTR_DIM if USE_EDGE_ATTR else None),
            beta=True, dropout=0.10
        )
        self.ln1 = nn.LayerNorm(h1 * 8)
        self.ln2 = nn.LayerNorm(h2 * 8)
        self.ln3 = nn.LayerNorm(h3 * 4)
        self.pool = GlobalAttention(gate_nn=nn.Linear(h3 * 4, 1))
        self.fc   = nn.Linear(h3 * 4, output_dim)
        self.act  = nn.ELU()

    def forward(self, data):
        x, ei = data.x, data.edge_index
        ea = getattr(data, "edge_attr", None) if USE_EDGE_ATTR else None
        x = self.conv1(x, ei, edge_attr=ea); x = self.ln1(self.act(x))
        x = self.conv2(x, ei, edge_attr=ea); x = self.ln2(self.act(x))
        x = self.conv3(x, ei, edge_attr=ea); x = self.ln3(self.act(x))
        g = self.pool(x, data.batch)
        return self.fc(g)

# ---------------------------------------
# Metrics helpers
# ---------------------------------------
def count_params_m(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6

def estimate_gflops_per_sample(n_nodes: int, n_edges: int, in_ch: int,
                               widths: list, heads: int, edge_attr_dim: int,
                               num_classes: int) -> float:
    flops = 0.0
    fin = in_ch
    H = heads
    for w in widths:
        mac_linears  = 3.0 * n_nodes * fin * (H * w)
        mac_edge     = n_edges * edge_attr_dim * (H * w) if edge_attr_dim > 0 else 0.0
        mac_dot      = n_edges * (H * w)
        mac_weighted = n_edges * (H * w)
        flops += 2.0 * (mac_linears + mac_edge + mac_dot + mac_weighted)
        fin = w
    last_dim = widths[-1] if widths else in_ch
    flops += 2.0 * (2.0 * n_nodes * last_dim)    # attention pooling (gate+sum)
    flops += 2.0 * (last_dim * num_classes)      # classifier
    return flops / 1e9

@torch.no_grad()
def measure_infer_ms_and_mem_per_sample(model: nn.Module, loader: DataLoader, device,
                                        warmup_batches=3, max_batches=20):
    model.eval()
    total_s, total_n = 0.0, 0
    peak_per_sample_bytes = 0.0

    it = iter(loader)
    for _ in range(min(warmup_batches, len(loader))):
        try:
            data = next(it).to(device)
        except StopIteration:
            break
        _ = model(data)

    counted = 0
    for data in loader:
        if counted >= max_batches:
            break
        n_graphs = data.num_graphs
        data = data.to(device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            base_alloc = torch.cuda.memory_allocated(device)

        t0 = time.perf_counter()
        _ = model(data)
        t1 = time.perf_counter()

        if device.type == "cuda":
            peak_alloc = torch.cuda.max_memory_allocated(device)
            extra = max(0, peak_alloc - base_alloc)
            per_sample = extra / max(1, n_graphs)
            peak_per_sample_bytes = max(peak_per_sample_bytes, per_sample)

        total_s += (t1 - t0)
        total_n += n_graphs
        counted += 1

    ms_per_sample = (total_s / max(1, total_n)) * 1000.0
    mem_mb_per_sample = (peak_per_sample_bytes / (1024.0**2)) if device.type == "cuda" else None
    return ms_per_sample, mem_mb_per_sample

# ---------------------------------------
# Train / Eval / Dump predictions
# ---------------------------------------
def train_one_epoch(model, loader, optim, crit, device, clip_grad=1.0):
    model.train()
    total = 0.0
    for data in loader:
        data = data.to(device)
        optim.zero_grad(set_to_none=True)
        out = model(data)
        loss = crit(out, data.y)
        if not torch.isfinite(loss):
            print(">> WARNING: non-finite loss; skipping batch")
            continue
        loss.backward()
        if clip_grad is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
        optim.step()
        total += float(loss.item())
    return total / max(1, len(loader))

@torch.no_grad()
def evaluate(model, loader, device, num_classes=NUM_CLASSES):
    model.eval()
    preds, labels = [], []
    for data in loader:
        data = data.to(device)
        out = model(data)
        preds.append(out.argmax(dim=1).cpu().numpy())
        labels.append(data.y.cpu().numpy())
    if not labels:
        zero_cm = np.zeros((num_classes, num_classes), dtype=int)
        return 0.0, 0.0, 0.0, zero_cm
    y_true = np.concatenate(labels)
    y_pred = np.concatenate(preds)
    micro_acc = accuracy_score(y_true, y_pred)

    cm  = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    per_class_tot = cm.sum(axis=1).astype(np.float64) + 1e-12
    per_class_acc = np.diag(cm) / per_class_tot
    macro_acc = float(np.mean(per_class_acc))
    macro_f1  = float(f1_score(y_true, y_pred, average='macro'))
    return micro_acc, macro_acc, macro_f1, cm

@torch.no_grad()
def predict_and_dump(model, loader, device, out_csv: str):
    model.eval()
    rows = []
    for data in loader:
        fnames = data.fname if isinstance(data.fname, (list, tuple)) else [data.fname]
        data = data.to(device)
        logits = model(data)                        # [B, C]
        probs  = F.softmax(logits, dim=1).cpu().numpy()
        pred   = np.argmax(probs, axis=1)
        y_true = data.y.cpu().numpy()
        B = logits.size(0)
        for b in range(B):
            row = {
                "filename": fnames[b] if b < len(fnames) else f"sample_{b}",
                "true_label": int(y_true[b]),
                "pred_label": int(pred[b])
            }
            for c in range(NUM_CLASSES):
                row[f"prob_{c}"] = float(probs[b, c])
            rows.append(row)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[Saved] Validation predictions → {out_csv}")

# ---------------------------------------
# Main
# ---------------------------------------
def main():
    # ---- Build datasets
    print("Loading DGW CSVs...")
    train_graphs = build_graphs_from_csv(train_dir)
    val_graphs   = build_graphs_from_csv(val_dir)

    # ---- DataLoaders (NO sampler)
    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True,
                              pin_memory=(DEVICE.type == "cuda"))
    val_loader   = DataLoader(val_graphs, batch_size=BATCH_SIZE, shuffle=False,
                              pin_memory=(DEVICE.type == "cuda"))

    # ---- Class-balanced focal loss
    counts = get_class_counts(train_graphs, NUM_CLASSES)
    CB_BETA = 1.0 - 1.0 / float(counts.sum())
    cb_w = class_balanced_weights(counts, beta=CB_BETA)
    print(f"CB_BETA used: {CB_BETA:.6f}")
    print("CB weights:", [float(f"{w:.6f}") for w in cb_w])
    class_weights_tensor = torch.tensor(cb_w, dtype=torch.float32, device=DEVICE)

    model = TransformerNet(num_node_features=3, output_dim=NUM_CLASSES).to(DEVICE)
    optimz = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched  = torch.optim.lr_scheduler.StepLR(optimz, step_size=STEP_SIZE, gamma=GAMMA)
    crit   = CBFocalLoss(class_weights=class_weights_tensor,
                         gamma=FOCAL_GAMMA,
                         label_smoothing=LABEL_SMOOTHING)

    # ---- Static metrics
    params_m = count_params_m(model)
    sample   = train_graphs[0]
    n_nodes  = sample.x.size(0)
    n_edges  = sample.edge_index.size(1)
    widths   = [512, 256, 64]  # from (h1*8, h2*8, h3*4)
    gflops   = estimate_gflops_per_sample(
        n_nodes, n_edges, in_ch=3,
        widths=widths, heads=HEADS,
        edge_attr_dim=(EDGE_ATTR_DIM if USE_EDGE_ATTR else 0),
        num_classes=NUM_CLASSES
    )
    print(f"Params={params_m:.3f}M | estGFLOPs/sample={gflops:.3f} | nodes={n_nodes} | edges={n_edges}")

    # ---- Train
    t_train_start = time.time()
    for ep in range(1, EPOCHS + 1):
        if DEVICE.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()

        loss = train_one_epoch(model, train_loader, optimz, crit, DEVICE, clip_grad=1.0)
        sched.step()

        if DEVICE.type == "cuda": torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        print(f"[Epoch {ep:03d}/{EPOCHS}] loss={loss:.4f}  time={dt:.2f}s  lr={optimz.param_groups[0]['lr']:.2e}",
              flush=True)
    train_time_s = time.time() - t_train_start
    print(f"Total training time: {train_time_s:.2f}s")

    # ---- Save model
    torch.save(model.state_dict(), SAVED_MODEL)
    print(f"[Saved] Model → {SAVED_MODEL}")

    # ---- Validation metrics
    if DEVICE.type == "cuda": torch.cuda.synchronize()
    t_val0 = time.perf_counter()
    micro_acc, macro_acc, macro_f1, cm = evaluate(model, val_loader, DEVICE, num_classes=NUM_CLASSES)
    if DEVICE.type == "cuda": torch.cuda.synchronize()
    val_total_time = time.perf_counter() - t_val0
    print(f"Validation total time: {val_total_time:.2f}s")
    print(f"Micro Accuracy: {micro_acc:.4f} | Macro Accuracy: {macro_acc:.4f} | Macro-F1: {macro_f1:.4f}")
    print("Confusion Matrix:\n", cm)

    # ---- Inference ms/sample + peak memory/sample (CUDA)
    infer_ms, mem_mb = measure_infer_ms_and_mem_per_sample(model, val_loader, DEVICE,
                                                           warmup_batches=3, max_batches=20)
    mem_str = (f"{mem_mb:.2f} MB/sample" if mem_mb is not None else "N/A (CPU)")
    print(f"Inference: {infer_ms:.2f} ms/sample | Peak additional memory: {mem_str}")

    # ---- Dump per-sample predictions (with filenames)
    predict_and_dump(model, val_loader, DEVICE, VAL_PRED_CSV)

if __name__ == "__main__":
    main()
