import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

from torch.cuda.amp import autocast, GradScaler
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import TransformerConv, GlobalAttention
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold

# ==============================
# Config (RGB-only, complete graph)
# ==============================
DATA_DIR = r"F:\DMD\S6_face_RGB"
NUM_SUBJECTS = 15
NUM_CLASSES  = 9
EPOCHS       = 50
BATCH_SIZE   = 2            # complete graph is heavy; start small
LR           = 1e-3
STEP_SIZE    = 20
GAMMA        = 0.5
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HEADS        = 8
LAYERS       = 3
USE_AMP      = True

# Complete-graph options
DIRECTED_DUPLICATE = False  # if True: add (j,i) for each (i,j) → doubles E & memory

CB_BETA         = 0.995   # set to None for plain focal
FOCAL_GAMMA     = 1.5
LABEL_SMOOTHING = 0.05
EPS             = 1e-8

# ==============================
# Loss: Class-Balanced Focal (with optional label smoothing)
# ==============================
def make_cb_focal_criterion(class_counts, num_classes=NUM_CLASSES, beta=CB_BETA,
                            gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTHING,
                            eps=EPS, device=None):
    counts = torch.as_tensor(class_counts, dtype=torch.float32)
    if device is not None:
        counts = counts.to(device)

    if beta is not None:
        effective_num = 1.0 - torch.pow(torch.as_tensor(beta), counts)
        weights = (1.0 - beta) / (effective_num + eps)
    else:
        weights = torch.ones_like(counts)

    # normalize to mean 1.0 (stable scale)
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
# Model (same as your baseline; no edge attrs here)
# ==============================
class TransformerNet(nn.Module):
    def __init__(self, num_node_features, output_dim):
        super().__init__()
        head_dim1 = 64
        head_dim2 = 32
        head_dim3 = 16
        self.conv1 = TransformerConv(num_node_features, head_dim1 * 8, heads=HEADS, concat=False)
        self.conv2 = TransformerConv(head_dim1 * 8, head_dim2 * 8, heads=HEADS, concat=False)
        self.conv3 = TransformerConv(head_dim2 * 8, head_dim3 * 4, heads=HEADS, concat=False)
        self.pool  = GlobalAttention(gate_nn=nn.Linear(head_dim3 * 4, 1))
        self.fc    = nn.Linear(head_dim3 * 4, output_dim)
        self.act   = nn.ELU()
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.act(self.conv1(x, edge_index))
        x = self.act(self.conv2(x, edge_index))
        x = self.act(self.conv3(x, edge_index))
        g = self.pool(x, data.batch)
        return self.fc(g)

# ==============================
# Complete graph (precomputed once)
# ==============================
def complete_graph_edge_index(n_nodes: int, directed_duplicate: bool = False) -> torch.Tensor:
    # edges for i<j (upper triangle), shape [2, E]
    i, j = np.triu_indices(n_nodes, k=1)
    ei = np.vstack([i, j]).astype(np.int64)
    if directed_duplicate:
        rev = np.vstack([j, i]).astype(np.int64)
        ei = np.hstack([ei, rev])
    return torch.from_numpy(ei)

# ==============================
# Build RGB graphs using the complete edge_index
# ==============================
def _to_zero_based(z):
    """Accept 0..8 or 1..9 → return 0..8; else raise."""
    z = int(z)
    if 0 <= z <= 8: return z
    if 1 <= z <= 9: return z - 1
    raise ValueError(f"Unexpected label (not in 0..8 or 1..9): {z}")

def build_graphs_complete(df_landmarks, edge_index):
    data_list = []
    for _, row in df_landmarks.iterrows():
        coords = row.iloc[3:].to_numpy(dtype=np.float32).reshape(-1, 3)  # [478,3]
        gaze_raw = int(row["Gaze Zone Number"])
        gaze     = _to_zero_based(gaze_raw)  # normalize to 0..8
        x  = torch.from_numpy(coords)
        y  = torch.tensor([gaze], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)
    return data_list

def load_rgb(csv_path, edge_index):
    df = pd.read_csv(csv_path)
    return build_graphs_complete(df, edge_index)

# ==============================
# Metrics: params, GFLOPs, times
# ==============================
def count_params_m(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6

def estimate_gflops_per_sample(n_nodes: int, n_edges: int, in_ch: int,
                               widths: list, heads: int, num_classes: int) -> float:
    """
    Coarse per-sample forward GFLOPs (no edge attrs):
      - Q/K/V projections
      - attention dot + (attn * V) per edge
      - GlobalAttention pooling
      - linear classifier
    Assumes concat=False; MACs -> FLOPs (×2).
    """
    flops = 0.0
    fin = in_ch
    H = heads
    for w in widths:
        mac_linears  = 3.0 * n_nodes * fin * (H * w)
        mac_dot      = n_edges * (H * w)
        mac_weighted = n_edges * (H * w)
        mac_layer = mac_linears + mac_dot + mac_weighted
        flops += 2.0 * mac_layer
        fin = w
    last_dim = widths[-1] if widths else in_ch
    mac_pool = 2.0 * n_nodes * last_dim
    flops += 2.0 * mac_pool
    flops += 2.0 * (last_dim * num_classes)
    return flops / 1e9

@torch.no_grad()
def measure_infer_ms_per_sample(model: nn.Module, loader: DataLoader, device, warmup_batches=5, max_batches=10):
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
def train_one_epoch(model, loader, optim, crit, device, scaler: GradScaler, amp: bool):
    model.train()
    total = 0.0
    for data in loader:
        data = data.to(device)
        optim.zero_grad(set_to_none=True)
        with autocast(enabled=(amp and device.type == "cuda")):
            out = model(data)
            loss = crit(out, data.y)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
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
# Main (RGB-only, complete graph)
# ==============================
if __name__ == "__main__":
    device = DEVICE

    # Precompute complete edge_index once for N=478
    N_NODES = 478
    edge_index = complete_graph_edge_index(N_NODES, directed_duplicate=DIRECTED_DUPLICATE).long()

    subjects = list(range(1, NUM_SUBJECTS + 1))
    kf = KFold(n_splits=NUM_SUBJECTS, shuffle=False)

    for fold, (train_idx, test_idx) in enumerate(kf.split(subjects), start=1):
        print(f"\n===== Fold {fold}/{NUM_SUBJECTS} (Complete graph, RGB-only, H={HEADS}, L={LAYERS}, "
              f"DirectedDup={DIRECTED_DUPLICATE}) =====")

        # --- Datasets (RGB only; reuse the same complete edge_index)
        train_graphs, test_graphs = [], []
        for i in train_idx:
            csv_rgb = os.path.join(DATA_DIR, str(subjects[i]), "Landmark_RGB_Valid.csv")
            train_graphs.extend(load_rgb(csv_rgb, edge_index))
        for i in test_idx:
            csv_rgb = os.path.join(DATA_DIR, str(subjects[i]), "Landmark_RGB_Valid.csv")
            test_graphs.extend(load_rgb(csv_rgb, edge_index))

        train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
        test_loader  = DataLoader(test_graphs,  batch_size=BATCH_SIZE, shuffle=False)

        # --- Model / Opt / Sched / AMP
        model = TransformerNet(num_node_features=3, output_dim=NUM_CLASSES).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=LR)
        sched = torch.optim.lr_scheduler.StepLR(optim, step_size=STEP_SIZE, gamma=GAMMA)

        # ======= Build CB-Focal criterion using TRAIN fold label counts =======
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

        scaler = GradScaler(enabled=(USE_AMP and device.type == "cuda"))

        # --- Static metrics (params, GFLOPs) from one sample
        params_m = count_params_m(model)
        widths   = [512, 256, 64]  # 3 layers, concat=False
        sample   = train_graphs[0]
        n_nodes  = sample.x.size(0)                               # 478
        n_edges  = sample.edge_index.size(1)                      # ≈ 478*477/2 if no duplicates
        gflops   = estimate_gflops_per_sample(n_nodes, n_edges, in_ch=3,
                                              widths=widths, heads=HEADS,
                                              num_classes=NUM_CLASSES)

        print(f"[Fold {fold}] nodes={n_nodes}  edges/graph={n_edges}  "
              f"params={params_m:.3f}M  estGFLOPs/sample={gflops:.3f}")

        # --- Train with per-epoch timing
        t_train_start = time.time()
        for ep in range(1, EPOCHS + 1):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_ep_start = time.perf_counter()

            loss = train_one_epoch(model, train_loader, optim, crit, device, scaler, USE_AMP)
            sched.step()

            if device.type == "cuda":
                torch.cuda.synchronize()
            t_ep_end = time.perf_counter()
            dt = t_ep_end - t_ep_start
            curr_lr = optim.param_groups[0]["lr"]
            print(f"[Fold {fold}] epoch {ep:03d}/{EPOCHS}  loss={loss:.4f}  time={dt:.1f}s  lr={curr_lr:.2e}",
                  flush=True)
        train_time_s = time.time() - t_train_start

        # --- Inference time
        infer_ms = measure_infer_ms_per_sample(model, test_loader, device, warmup_batches=5, max_batches=10)

        # --- Evaluate (once per fold)
        acc, cm = evaluate(model, test_loader, device)
        print(f"[Fold {fold}] Params={params_m:.3f}M  GFLOPs/sample={gflops:.3f}  "
              f"TrainTime={train_time_s:.1f}s  Infer={infer_ms:.2f} ms/sample")
        print(f"[Fold {fold}] Accuracy={acc:.4f}")
        print(f"[Fold {fold}] Confusion Matrix:\n{cm}")
