import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import TransformerConv, GlobalAttention
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold

# ==============================
# Config (RGB-only, kNN + edge attrs, STABLE)
# ==============================
DATA_DIR = r"...\DMD"
NUM_SUBJECTS = 15
NUM_CLASSES  = 9
EPOCHS       = 50
BATCH_SIZE   = 4
LR           = 1e-3
WEIGHT_DECAY = 1e-4
STEP_SIZE    = 20
GAMMA        = 0.5
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

K_NEIGHBORS       = 8
SYMMETRIC_EDGES   = False   # directed kNN edges (i -> k neighbors)
USE_EDGE_ATTR     = True
EDGE_ATTR_DIM     = 4       # (ux, uy, uz, log1p(dist))
HEADS             = 8
LAYERS            = 3
USE_AMP           = False    # keep off for this ablation

EPS = 1e-6

CB_BETA         = 0.995     # set to None for plain focal (no class balancing)
FOCAL_GAMMA     = 1.5
LABEL_SMOOTHING = 0.05

# ==============================
# Class-Balanced Focal Loss (+ optional label smoothing)
# ==============================
def make_cb_focal_criterion(class_counts, num_classes=NUM_CLASSES, beta=CB_BETA,
                            gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTHING,
                            eps=1e-8, device=None):
    counts = torch.as_tensor(class_counts, dtype=torch.float32)
    if device is not None:
        counts = counts.to(device)

    if beta is not None:
        effective_num = 1.0 - torch.pow(torch.as_tensor(beta, device=counts.device), counts)
        weights = (1.0 - beta) / (effective_num + eps)
    else:
        weights = torch.ones_like(counts)

    # normalize to mean 1.0 for stable scaling
    weights = weights * (num_classes / (weights.sum() + eps))

    def criterion(logits, targets):
        # logits: [B, C], targets: [B]
        probs = torch.softmax(logits, dim=1).clamp_min(eps)

        with torch.no_grad():
            y_true = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1.0)
            if label_smoothing and label_smoothing > 0:
                y_true = (1.0 - label_smoothing) * y_true + (label_smoothing / num_classes)

        p_t = (probs * y_true).sum(dim=1).clamp_min(eps)   # [B]
        focal = (1.0 - p_t).pow(gamma)                     # [B]

        alpha = weights.to(logits.device)                  # [C]
        ce_per_class = -y_true * torch.log(probs) * alpha  # [B, C]
        ce = ce_per_class.sum(dim=1)                       # [B]

        return (focal * ce).mean()

    return criterion

# ==============================
# Model (TransformerConv with edge_attr + LayerNorm + dropout)
# ==============================
class TransformerNet(nn.Module):
    def __init__(self, num_node_features, output_dim):
        super().__init__()
        h1, h2, h3 = 64, 32, 16
        self.conv1 = TransformerConv(num_node_features, h1 * 8, heads=HEADS,
                                     concat=False, edge_dim=EDGE_ATTR_DIM,
                                     beta=True, dropout=0.10)
        self.conv2 = TransformerConv(h1 * 8,   h2 * 8, heads=HEADS,
                                     concat=False, edge_dim=EDGE_ATTR_DIM,
                                     beta=True, dropout=0.10)
        self.conv3 = TransformerConv(h2 * 8,   h3 * 4, heads=HEADS,
                                     concat=False, edge_dim=EDGE_ATTR_DIM,
                                     beta=True, dropout=0.10)
        self.ln1 = nn.LayerNorm(h1 * 8)
        self.ln2 = nn.LayerNorm(h2 * 8)
        self.ln3 = nn.LayerNorm(h3 * 4)
        self.pool  = GlobalAttention(gate_nn=nn.Linear(h3 * 4, 1))
        self.fc    = nn.Linear(h3 * 4, output_dim)
        self.act   = nn.ELU()

    def forward(self, data):
        x, ei, ea = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, ei, edge_attr=ea); x = self.ln1(self.act(x))
        x = self.conv2(x, ei, edge_attr=ea); x = self.ln2(self.act(x))
        x = self.conv3(x, ei, edge_attr=ea); x = self.ln3(self.act(x))
        g = self.pool(x, data.batch)
        return self.fc(g)

# ==============================
# Geometry utils (normalize coords, stable edge attrs)
# ==============================
def normalize_coords(coords_np):
    """Center to zero-mean and scale by RMS radius for scale/translation invariance."""
    c = coords_np.astype(np.float32)
    center = c.mean(axis=0, keepdims=True)
    c0 = c - center
    rms = np.sqrt((c0**2).sum(axis=1).mean()) + EPS
    return c0 / rms

def knn_graph_with_attrs(coords_np, k=8, symmetric=False):
    """
    Build kNN on normalized coords; edge_attr = [unit_vec(3), log1p(dist)].
    """
    c = normalize_coords(coords_np)         # [N,3]
    N = c.shape[0]
    # pairwise distances
    diff = c[:, None, :] - c[None, :, :]
    dist2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dist2, np.inf)
    nbr_idx = np.argpartition(dist2, kth=min(k, N-1), axis=1)[:, :k]  # [N,k]

    rows, cols = [], []
    for i in range(N):
        js = nbr_idx[i]
        rows.extend([i] * len(js))
        cols.extend(js.tolist())

    if symmetric:
        rows_sym = rows + cols
        cols_sym = cols + rows
        rows, cols = rows_sym, cols_sym

    edge_index = np.vstack([rows, cols]).astype(np.int64)  # [2,E]
    src = c[edge_index[0]]    # [E,3]
    dst = c[edge_index[1]]    # [E,3]
    vec = dst - src
    dist = np.linalg.norm(vec, axis=1, keepdims=True)      # [E,1]
    unit = vec / np.clip(dist, EPS, None)                  # safe unit vector
    logd = np.log1p(dist)                                  # stable distance channel
    edge_attr = np.concatenate([unit, logd], axis=1).astype(np.float32)  # [E,4]
    return torch.from_numpy(edge_index), torch.from_numpy(edge_attr)

# ==============================
# Data utils (RGB only)
# ==============================
def _to_zero_based(z):
    """Accept 0..8 or 1..9 -> return 0..8; else raise."""
    z = int(z)
    if 0 <= z <= 8: return z
    if 1 <= z <= 9: return z - 1
    raise ValueError(f"Unexpected label (not in 0..8 or 1..9): {z}")

def build_graphs_knn(df_landmarks, k=K_NEIGHBORS):
    data_list = []
    for _, row in df_landmarks.iterrows():
        coords = row.iloc[3:].to_numpy(dtype=np.float32).reshape(-1, 3)  # [478,3]
        gaze_raw = int(row["Gaze Zone Number"])
        gaze     = _to_zero_based(gaze_raw)  # normalize to 0..8

        ei, ea = knn_graph_with_attrs(coords, k=k, symmetric=SYMMETRIC_EDGES)
        x = torch.from_numpy(normalize_coords(coords))  # normalized node features
        y = torch.tensor([gaze], dtype=torch.long)
        data = Data(x=x, edge_index=ei.long(), edge_attr=ea, y=y)
        data_list.append(data)
    return data_list

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    return build_graphs_knn(df, k=K_NEIGHBORS)

# ==============================
# Metrics: params, GFLOPs, times
# ==============================
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
    flops += 2.0 * (2.0 * n_nodes * last_dim)    # pool
    flops += 2.0 * (last_dim * num_classes)      # classifier
    return flops / 1e9

@torch.no_grad()
def measure_infer_ms_per_sample(model: nn.Module, loader: DataLoader, device, warmup_batches=5, max_batches=20):
    model.eval()
    # warmup
    it = iter(loader)
    for _ in range(min(warmup_batches, len(loader))):
        try:
            data = next(it).to(device)
        except StopIteration:
            break
        _ = model(data)
    # timed
    total_s, total_n = 0.0, 0
    if device.type == "cuda":
        torch.cuda.synchronize()
    counted = 0
    for data in loader:
        if counted >= max_batches:
            break
        data = data.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(data)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        total_s += (t1 - t0)
        total_n += data.num_graphs
        counted += 1
    return (total_s / max(1, total_n)) * 1000.0

# ==============================
# Train / Eval
# ==============================
def train_one_epoch(model, loader, optim, crit, device, clip_grad=1.0):
    model.train()
    total = 0.0
    for data in loader:
        data = data.to(device)
        optim.zero_grad(set_to_none=True)
        out = model(data)
        loss = crit(out, data.y)

        if not torch.isfinite(loss):
            print(">> WARNING: non-finite loss detected; skipping batch")
            continue

        loss.backward()
        if clip_grad is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
        optim.step()
        total += float(loss.item())
    return total / max(1, len(loader))

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    for data in loader:
        data = data.to(device)
        out = model(data)
        p = out.argmax(dim=1).cpu().numpy()
        y = data.y.cpu().numpy()
        preds.append(p); labels.append(y)
    if not labels:
        return 0.0, np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    y_true = np.concatenate(labels)
    y_pred = np.concatenate(preds)
    acc = accuracy_score(y_true, y_pred)
    cm  = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    return acc, cm

# ==============================
# Main (RGB-only, kNN + edge attrs, stable)
# ==============================
if __name__ == "__main__":
    device = DEVICE

    subjects = list(range(1, NUM_SUBJECTS + 1))
    kf = KFold(n_splits=NUM_SUBJECTS, shuffle=False)

    for fold, (train_idx, test_idx) in enumerate(kf.split(subjects), start=1):
        print(f"\n===== Fold {fold}/{NUM_SUBJECTS} (kNN k={K_NEIGHBORS}, edge_attr ON, RGB-only, H={HEADS}) =====")

        # --- Build datasets (RGB only)
        train_graphs, test_graphs = [], []
        for i in train_idx:
            csv_rgb = os.path.join(DATA_DIR, str(subjects[i]), "Landmark_RGB_Valid.csv")
            train_graphs.extend(load_data(csv_rgb))
        for i in test_idx:
            csv_rgb = os.path.join(DATA_DIR, str(subjects[i]), "Landmark_RGB_Valid.csv")
            test_graphs.extend(load_data(csv_rgb))

        train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
        test_loader  = DataLoader(test_graphs,  batch_size=BATCH_SIZE, shuffle=False)

        # --- Model / Opt / Sched
        model = TransformerNet(num_node_features=3, output_dim=NUM_CLASSES).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.StepLR(optim, step_size=STEP_SIZE, gamma=GAMMA)

        # ====== Build CB-Focal criterion using TRAIN-fold class counts ======
        counts = np.bincount([int(d.y.item()) for d in train_graphs], minlength=NUM_CLASSES)
        print("Train class counts:", counts.tolist())
        crit  = make_cb_focal_criterion(
            class_counts=counts,
            num_classes=NUM_CLASSES,
            beta=CB_BETA,
            gamma=FOCAL_GAMMA,
            label_smoothing=LABEL_SMOOTHING,
            device=device
        )

        # --- Static metrics (params, GFLOPs) from a kNN sample
        params_m = count_params_m(model)
        widths   = [512, 256, 64]  # L=3
        sample   = train_graphs[0]
        n_nodes  = sample.x.size(0)
        n_edges  = sample.edge_index.size(1)
        gflops   = estimate_gflops_per_sample(n_nodes, n_edges, in_ch=3,
                                              widths=widths, heads=HEADS,
                                              edge_attr_dim=(EDGE_ATTR_DIM if USE_EDGE_ATTR else 0),
                                              num_classes=NUM_CLASSES)
        print(f"[Fold {fold}] nodes={n_nodes}  edges/graph={n_edges}  "
              f"params={params_m:.3f}M  estGFLOPs/sample={gflops:.3f}")

        # --- Train with per-epoch timing
        t_train_start = time.time()
        for ep in range(1, EPOCHS + 1):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_ep_start = time.perf_counter()

            loss = train_one_epoch(model, train_loader, optim, crit, device, clip_grad=1.0)
            sched.step()

            if device.type == "cuda":
                torch.cuda.synchronize()
            t_ep_end = time.perf_counter()
            dt = t_ep_end - t_ep_start
            curr_lr = optim.param_groups[0]["lr"]
            print(f"[Fold {fold}] epoch {ep:03d}/{EPOCHS}  loss={loss:.4f}  time={dt:.1f}s  lr={curr_lr:.2e}", flush=True)
        train_time_s = time.time() - t_train_start

        # --- Inference time (ms/sample)
        infer_ms = measure_infer_ms_per_sample(model, test_loader, device, warmup_batches=5, max_batches=20)

        # --- Evaluate (once per fold)
        acc, cm = evaluate(model, test_loader, device)
        print(f"[Fold {fold}] Params={params_m:.3f}M  GFLOPs/sample={gflops:.3f}  "
              f"TrainTime={train_time_s:.1f}s  Infer={infer_ms:.2f} ms/sample")
        print(f"[Fold {fold}] Accuracy={acc:.4f}")
        print(f"[Fold {fold}] Confusion Matrix:\n{cm}")
