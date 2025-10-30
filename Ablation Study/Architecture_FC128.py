import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import networkx as nx

from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import TransformerConv, GlobalAttention
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold

# ==============================
# Config
# ==============================
DATA_DIR = r"...\DMD"   # RGB only
NUM_SUBJECTS = 15
NUM_CLASSES = 9
EPOCHS = 50
BATCH_SIZE = 32
LR = 1e-3
STEP_SIZE = 20
GAMMA = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# ==== config ====
HEADS = 8  # <-- ablation: number of attention heads

# ----------------------------
# TransformerNet (heads = 2)
# ----------------------------
class TransformerNet(nn.Module):
    """
    Ablation: tail (final layer output width) = 128 instead of 64
    - H = 8 heads, concat=False
    - Layer widths: 512 -> 256 -> 128
    - No edge attributes (baseline topology unchanged)
    """
    def __init__(self, num_node_features: int, output_dim: int, heads: int = 8, tail_dim: int = 128):
        super().__init__()
        assert tail_dim == 128, "This ablation sets the final node width to 128."

        head_dim1 = 64   # 64 * 8 = 512
        head_dim2 = 32   # 32 * 8 = 256

        # concat=False => output width == out_channels (not heads * out_channels)
        self.conv1 = TransformerConv(num_node_features, head_dim1 * heads, heads=heads, concat=False)  # 3 -> 512
        self.conv2 = TransformerConv(head_dim1 * heads, head_dim2 * heads, heads=heads, concat=False) # 512 -> 256
        self.conv3 = TransformerConv(head_dim2 * heads, tail_dim,            heads=heads, concat=False) # 256 -> 128

        self.att_pool = GlobalAttention(gate_nn=nn.Linear(tail_dim, 1))  # gate over 128-D nodes
        self.fc       = nn.Linear(tail_dim, output_dim)                  # 128 -> 9 classes
        self.act      = nn.ELU()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.act(self.conv1(x, edge_index))  # [N, 512]
        x = self.act(self.conv2(x, edge_index))  # [N, 256]
        x = self.act(self.conv3(x, edge_index))  # [N, 128]  <-- wider tail
        g = self.att_pool(x, batch)              # [B, 128]
        return self.fc(g)

# ==============================
# Data utils
# ==============================
def build_graphs(df_landmarks, edges):
    data_list = []
    for _, row in df_landmarks.iterrows():
        landmarks = row.iloc[3:].values.reshape(-1, 3).astype(np.float32)
        gaze_zone = int(row["Gaze Zone Number"])
        G = nx.Graph()
        for i, landmark in enumerate(landmarks):
            G.add_node(i, pos=landmark)
        G.add_edges_from(edges)
        x = torch.tensor([G.nodes[i]['pos'] for i in G.nodes()], dtype=torch.float32)
        edge_index = torch.tensor(list(G.edges)).t().contiguous().long()
        y = torch.tensor([gaze_zone], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)
    return data_list

def load_data(csv_path, edges):
    df = pd.read_csv(csv_path)
    return build_graphs(df, edges)

# ==============================
# Metrics: params, FLOPs, times
# ==============================
def count_params_m(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6

def estimate_gflops_per_sample(n_nodes: int, n_edges: int, in_ch: int, widths: list, heads: int, num_classes: int) -> float:
    """
    Transparent, coarse estimate for one forward pass/sample:
    - Q/K/V projections
    - attention dot + weighted sum per edge
    - global attention pooling
    - final FC
    Assumes concat=False; counts MACs and converts to FLOPs (×2).
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
    flops += 2.0 * (2.0 * n_nodes * last_dim)          # pool (gate + weighted sum)
    flops += 2.0 * (last_dim * num_classes)            # classifier
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
def train_one_epoch(model, loader, optim, crit, device):
    model.train()
    total = 0.0
    for data in loader:
        data = data.to(device)
        optim.zero_grad()
        out = model(data)
        loss = crit(out, data.y)
        loss.backward()
        optim.step()
        total += loss.item()
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
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    return acc, cm

# ==============================
# Main (RGB only, heads=16)
# ==============================
if __name__ == "__main__":
    device = DEVICE

    # Graph edges (your 998-ish defined edges)
    edges = [(468, node) for node in range(478) if node != 468]
    edges += [(473, node) for node in range(478) if node != 473]
    edges.extend([(471, 159), (159, 469), (469, 145), (145, 471), (476, 475), (475, 474), (474, 477), (477, 476),
                  (1, 33), (1, 173), (1, 162), (1, 263), (1, 398), (1, 368), (33, 246), (146, 161), (161, 160),
                  (160, 150), (150, 158), (158, 157), (157, 173), (173, 155), (155, 154), (154, 153), (153, 145),
                  (145, 144), (144, 163), (163, 7), (7, 33), (398, 384), (384, 385), (385, 386), (386, 387), (387, 388),
                  (388, 263), (263, 249), (249, 390), (390, 373), (373, 374), (374, 380), (380, 381), (381, 382),
                  (382, 398)])

    subjects = list(range(1, NUM_SUBJECTS + 1))
    kf = KFold(n_splits=NUM_SUBJECTS, shuffle=False)

    for fold, (train_idx, test_idx) in enumerate(kf.split(subjects), start=1):
        print(f"\n===== Fold {fold}/{NUM_SUBJECTS} (last layer=128, RGB-only) =====")

        # --- Build datasets (RGB only)
        train_graphs, test_graphs = [], []
        for i in train_idx:
            csv_rgb = os.path.join(DATA_DIR, str(subjects[i]), "Landmark_RGB_Valid.csv")
            train_graphs.extend(load_data(csv_rgb, edges))
        for i in test_idx:
            csv_rgb = os.path.join(DATA_DIR, str(subjects[i]), "Landmark_RGB_Valid.csv")
            test_graphs.extend(load_data(csv_rgb, edges))

        train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
        test_loader  = DataLoader(test_graphs,  batch_size=BATCH_SIZE, shuffle=False)

        # --- Model / Opt / Sched
        model = TransformerNet(num_node_features=3, output_dim=9, heads=8).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=LR)
        sched = torch.optim.lr_scheduler.StepLR(optim, step_size=STEP_SIZE, gamma=GAMMA)
        crit  = nn.CrossEntropyLoss()

        # --- Static metrics (params, FLOPs)
        params_m = count_params_m(model)
        # widths are 512 -> 256 -> 64 as defined by your head_dim scheme
        widths = [512, 256, 128]
        # take graph sizes from any sample (all have same topology)
        sample = train_graphs[0]
        n_nodes = sample.x.size(0)
        n_edges = sample.edge_index.size(1)
        gflops = estimate_gflops_per_sample(n_nodes, n_edges, in_ch=3, widths=widths, heads=16, num_classes=NUM_CLASSES)

        # --- Train (time it)
        t0 = time.time()
        for ep in range(1, EPOCHS + 1):
            # per-epoch timing (CUDA-safe)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            loss = train_one_epoch(model, train_loader, optim, crit, device)
            sched.step()

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            dt = t1 - t0

            curr_lr = optim.param_groups[0]["lr"]
            print(f"[Fold {fold}] epoch {ep:03d}/{EPOCHS}  loss={loss:.4f}  time={dt:.1f}s  lr={curr_lr:.2e}",
                  flush=True)
        train_time_s = time.time() - t0

        # --- Inference time
        infer_ms = measure_infer_ms_per_sample(model, test_loader, device, warmup_batches=5, max_batches=20)

        # --- Evaluate (once per fold)
        acc, cm = evaluate(model, test_loader, device)
        print(f"[Fold {fold}] Params={params_m:.3f}M  GFLOPs/sample={gflops:.3f}  "
              f"TrainTime={train_time_s:.1f}s  Infer={infer_ms:.2f} ms/sample")
        print(f"[Fold {fold}] Accuracy={acc:.4f}")
        print(f"[Fold {fold}] Confusion Matrix:\n{cm}")
