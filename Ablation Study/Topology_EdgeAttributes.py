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
# Config
# ==============================
DATA_DIR     = r"...\DMD"
NUM_SUBJECTS = 15
NUM_CLASSES  = 9
EPOCHS       = 50
BATCH_SIZE   = 4
LR           = 1e-3
WEIGHT_DECAY = 5e-4
STEP_SIZE    = 20
GAMMA        = 0.5
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

START_FOLD   = 1

# Topology: fixed edges (baseline) + edge attributes
SYMMETRIC_EDGES = False     # If True, add both (u,v) and (v,u)
USE_EDGE_ATTR   = True
EDGE_ATTR_DIM   = 4         # (ux, uy, uz, log1p(dist))

# GNN
HEADS   = 8
LAYERS  = 3
USE_AMP = False

EPS = 1e-6

# ==============================
# Model (TransformerConv + edge_attr)
# ==============================
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

# ==============================
# Geometry utils (normalize + edge_attr on FIXED edges)
# ==============================
def normalize_coords(coords_np: np.ndarray) -> np.ndarray:
    c = coords_np.astype(np.float32)
    center = c.mean(axis=0, keepdims=True)
    c0 = c - center
    rms = np.sqrt((c0**2).sum(axis=1).mean()) + EPS
    return c0 / rms

def build_edge_attr_for_fixed_edges(coords_np: np.ndarray, edges_list: list, symmetric: bool = False):
    undirected_edges = [(int(u), int(v)) for (u, v) in edges_list]
    rows, cols = [], []
    for u, v in undirected_edges:
        rows.append(u); cols.append(v)
        if symmetric:
            rows.append(v); cols.append(u)
    edge_index = np.vstack([rows, cols]).astype(np.int64)  # [2,E]
    src = coords_np[edge_index[0]]  # [E,3]
    dst = coords_np[edge_index[1]]  # [E,3]
    vec = dst - src
    dist = np.linalg.norm(vec, axis=1, keepdims=True)      # [E,1]
    unit = vec / np.clip(dist, EPS, None)                  # [E,3]
    logd = np.log1p(dist)                                  # [E,1]
    edge_attr = np.concatenate([unit, logd], axis=1).astype(np.float32)
    return torch.from_numpy(edge_index), torch.from_numpy(edge_attr)

# ==============================
# Data utils (RGB only, FIXED topology)
# ==============================
def build_graphs_fixed(df_landmarks, fixed_edges):
    data_list = []
    for _, row in df_landmarks.iterrows():
        coords = row.iloc[3:].to_numpy(dtype=np.float32).reshape(-1, 3)  # [N,3]
        gaze   = int(row["Gaze Zone Number"])

        coords_norm = normalize_coords(coords)                            # [N,3]
        x = torch.from_numpy(coords_norm)

        ei, ea = build_edge_attr_for_fixed_edges(coords_norm, fixed_edges, SYMMETRIC_EDGES)
        y = torch.tensor([gaze], dtype=torch.long)

        data = Data(x=x, edge_index=ei.long(), edge_attr=(ea if USE_EDGE_ATTR else None), y=y)
        data_list.append(data)
    return data_list

def load_data(csv_path, fixed_edges):
    df = pd.read_csv(csv_path)
    return build_graphs_fixed(df, fixed_edges)

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
    flops += 2.0 * (2.0 * n_nodes * last_dim)          # GlobalAttention
    flops += 2.0 * (last_dim * num_classes)            # classifier
    return flops / 1e9

@torch.no_grad()
def measure_infer_ms_per_sample(model: nn.Module, loader: DataLoader, device, warmup_batches=5, max_batches=20):
    model.eval()
    it = iter(loader)
    # warmup
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
            print(">> WARNING: non-finite loss; skipping batch")
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
# Fixed baseline edges (998)
# ==============================
def get_fixed_edges():
    edges = [(468, node) for node in range(478) if node != 468]
    edges += [(473, node) for node in range(478) if node != 473]
    edges.extend([
        (471, 159), (159, 469), (469, 145), (145, 471),
        (476, 475), (475, 474), (474, 477), (477, 476),
        (1, 33), (1, 173), (1, 162), (1, 263), (1, 398), (1, 368),
        (33, 246), (146, 161), (161, 160), (160, 150), (150, 158),
        (158, 157), (157, 173), (173, 155), (155, 154), (154, 153),
        (153, 145), (145, 144), (144, 163), (163, 7), (7, 33),
        (398, 384), (384, 385), (385, 386), (386, 387), (387, 388),
        (388, 263), (263, 249), (249, 390), (390, 373), (373, 374),
        (374, 380), (380, 381), (381, 382), (382, 398)
    ])
    return edges

# ==============================
# Main (RGB-only, FIXED topology + edge attrs)
# ==============================
if __name__ == "__main__":
    device = DEVICE
    FIXED_EDGES = get_fixed_edges()

    subjects = list(range(1, NUM_SUBJECTS + 1))  # [1..15]
    kf = KFold(n_splits=NUM_SUBJECTS, shuffle=False)

    print(f"Resuming cross-validation from fold {START_FOLD} to {NUM_SUBJECTS} (inclusive).")

    for fold, (train_idx, test_idx) in enumerate(kf.split(subjects), start=1):
        # Skip completed folds 1..START_FOLD-1
        if fold < START_FOLD:
            continue

        print(f"\n===== Fold {fold}/{NUM_SUBJECTS} (FIXED topology, edge_attr ON, RGB-only, H={HEADS}) =====")

        # --- Build datasets (RGB only)
        train_graphs, test_graphs = [], []
        for i in train_idx:
            csv_rgb = os.path.join(DATA_DIR, str(subjects[i]), "Landmark_RGB_Valid.csv")
            train_graphs.extend(load_data(csv_rgb, FIXED_EDGES))
        for i in test_idx:
            csv_rgb = os.path.join(DATA_DIR, str(subjects[i]), "Landmark_RGB_Valid.csv")
            test_graphs.extend(load_data(csv_rgb, FIXED_EDGES))

        train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
        test_loader  = DataLoader(test_graphs,  batch_size=BATCH_SIZE, shuffle=False)

        # --- Model / Opt / Sched
        model = TransformerNet(num_node_features=3, output_dim=NUM_CLASSES).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.StepLR(optim, step_size=STEP_SIZE, gamma=GAMMA)
        crit  = nn.CrossEntropyLoss()

        # --- Static metrics (Params, GFLOPs) from a sample
        params_m = count_params_m(model)
        sample   = train_graphs[0]
        n_nodes  = sample.x.size(0)
        n_edges  = sample.edge_index.size(1)
        widths   = [512, 256, 64]  # L=3
        gflops   = estimate_gflops_per_sample(
            n_nodes, n_edges, in_ch=3, widths=widths, heads=HEADS,
            edge_attr_dim=(EDGE_ATTR_DIM if USE_EDGE_ATTR else 0),
            num_classes=NUM_CLASSES
        )
        print(f"[Fold {fold}] nodes={n_nodes}  edges/graph={n_edges}  "
              f"params={params_m:.3f}M  estGFLOPs/sample={gflops:.3f}")

        # --- Train (timed)
        t_train_start = time.time()
        for ep in range(1, EPOCHS + 1):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            loss = train_one_epoch(model, train_loader, optim, crit, device, clip_grad=1.0)
            sched.step()

            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = (time.perf_counter() - t0)
            print(f"[Fold {fold}] epoch {ep:03d}/{EPOCHS}  loss={loss:.4f}  time={dt:.1f}s  lr={optim.param_groups[0]['lr']:.2e}",
                  flush=True)
        train_time_s = time.time() - t_train_start

        # --- Inference timing
        infer_ms = measure_infer_ms_per_sample(model, test_loader, device, warmup_batches=5, max_batches=20)

        # --- Evaluate
        acc, cm = evaluate(model, test_loader, device)
        print(f"[Fold {fold}] Params={params_m:.3f}M  GFLOPs/sample={gflops:.3f}  "
              f"TrainTime={train_time_s:.1f}s  Infer={infer_ms:.2f} ms/sample")
        print(f"[Fold {fold}] Accuracy={acc:.4f}")
        print(f"[Fold {fold}] Confusion Matrix:\n{cm}")
