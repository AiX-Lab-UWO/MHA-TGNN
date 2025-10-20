# ==========================================
# Cross-dataset (Label-Aligned 7-Class) Validation
#  - Model: TransformerConv + fixed topology + edge attributes ONLY
#  - A) Train on ALL DMD -> Test on DGW (flipped) VAL
#  - B) Train on DGW (flipped) TRAIN -> Test on each DMD subject; average metrics
# ==========================================
import os, time, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import TransformerConv, AttentionalAggregation as GlobalAttention

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
)

# -----------------------------
# Paths
# -----------------------------
DMD_DIR          = r"...\DMD"  # expects subdirs 1..15, each has Landmark_RGB_Valid.csv
DGW_TRAIN_CSV    = r"...\DGW\DGW_Dataset\flipped\train_flipped_landmarks.csv"
DGW_VAL_CSV      = r"...\DGW\DGW_Dataset\flipped\val_flipped_landmarks.csv"

# -----------------------------
# Config
# -----------------------------
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS         = 50
BATCH_SIZE     = 32
LR             = 1e-3
STEP_SIZE      = 20
GAMMA          = 0.5
HEADS          = 8
EPS            = 1e-6
USE_EDGE_ATTR  = True
EDGE_ATTR_DIM  = 4   # [ux, uy, uz, log1p(dist)]
SYMMETRIC_EDGES = False  # keep the directed prior as in your baseline
NUM_SUBJECTS   = 15

# ========= Unified 7-class mapping (your rules) =========
# We convert both datasets to a shared label space {0..6}:
# U0 := DMD zone 3                <-> DGW zones {1,2} merged
# U1 := DMD zone 9                <-> DGW zone 3
# U2 := DMD zone 8                <-> DGW zone 4
# U3 := DMD zone 4                <-> DGW zone 8
# U4 := DMD zone 5                <-> DGW zones {5,6} merged
# U5 := DMD zones {6,7} merged    <-> DGW zone 7
# U6 := DMD zones {1,2} merged    <-> DGW zone 9

def _to_1based(z):
    """Accept 0..8 or 1..9 -> return 1..9; otherwise raise."""
    z = int(z)
    if 0 <= z <= 8:  # 0-based
        return z + 1
    if 1 <= z <= 9:
        return z
    raise ValueError(f"Unexpected label (not in 0..8 or 1..9): {z}")

def map_dgw_1to9_to_unified(z_1to9: int) -> int:
    z = _to_1based(z_1to9)
    if z in (1, 2): return 0  # -> U0 (DMD 3)
    if z == 3:      return 1  # -> U1 (DMD 9)
    if z == 4:      return 2  # -> U2 (DMD 8)
    if z == 8:      return 3  # -> U3 (DMD 4)
    if z in (5, 6): return 4  # -> U4 (DMD 5)
    if z == 7:      return 5  # -> U5 (DMD 6/7)
    if z == 9:      return 6  # -> U6 (DMD 1/2)
    raise ValueError(f"DGW label out of 1..9: {z}")

def map_dmd_to_unified(z_dmd: int) -> int:
    z = _to_1based(z_dmd)  # DMD sometimes 0..8 in CSV
    if z == 3:      return 0  # -> U0
    if z == 9:      return 1  # -> U1
    if z == 8:      return 2  # -> U2
    if z == 4:      return 3  # -> U3
    if z == 5:      return 4  # -> U4
    if z in (6, 7): return 5  # -> U5
    if z in (1, 2): return 6  # -> U6
    raise ValueError(f"DMD label not recognized: {z}")

NUM_CLASSES = 7  # unified

# -----------------------------
# Model
# -----------------------------
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

# -----------------------------
# Fixed edges (same as baseline)
# -----------------------------
def get_fixed_edges(n_nodes=478):
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
    rows, cols = [], []
    for (u, v) in edges:
        rows.append(u); cols.append(v)
        if SYMMETRIC_EDGES:
            rows.append(v); cols.append(u)
    ei = np.vstack([rows, cols]).astype(np.int64)
    return torch.from_numpy(ei)

# -----------------------------
# Geometry & edge_attr
# -----------------------------
def normalize_coords(coords_np: np.ndarray) -> np.ndarray:
    c = coords_np.astype(np.float32)
    center = c.mean(axis=0, keepdims=True)
    c0 = c - center
    rms = np.sqrt((c0**2).sum(axis=1).mean()) + EPS
    return c0 / rms

def build_edge_attr(coords_norm: np.ndarray, edge_index_np: np.ndarray) -> torch.Tensor:
    src = coords_norm[edge_index_np[0]]   # [E,3]
    dst = coords_norm[edge_index_np[1]]   # [E,3]
    vec = dst - src
    dist = np.linalg.norm(vec, axis=1, keepdims=True)
    unit = vec / np.clip(dist, EPS, None)
    logd = np.log1p(dist)
    ea = np.concatenate([unit, logd], axis=1).astype(np.float32)  # [E,4]
    return torch.from_numpy(ea)

# -----------------------------
# DGW CSV -> graphs (flipped sets you created)
#   Schema: filename, x0,y0,z0,...,x477,y477,z477, Label (1..9)
# -----------------------------
def build_graphs_from_dgw_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    graphs = []
    for _, row in df.iterrows():
        label_1to9 = int(row.iloc[-1])
        label_u = map_dgw_1to9_to_unified(label_1to9)

        vals = row.iloc[1:-1].to_numpy(dtype=np.float32)
        L = (len(vals) // 3) * 3
        coords = vals[:L].reshape(-1, 3)
        coords_norm = normalize_coords(coords)
        n = coords_norm.shape[0]
        ei = get_fixed_edges(n)
        ea = build_edge_attr(coords_norm, ei.numpy()) if USE_EDGE_ATTR else None

        data = Data(
            x=torch.from_numpy(coords_norm),
            edge_index=ei.long(),
            edge_attr=ea,
            y=torch.tensor([label_u], dtype=torch.long)
        )
        graphs.append(data)
    return graphs

# -----------------------------
# DMD CSV -> graphs
#   Schema: first 3 meta cols, landmarks from col 3 on, label column "Gaze Zone Number"
#   Labels may be 0..8 or 1..9; mapping handles both.
# -----------------------------
def build_graphs_from_dmd_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    graphs = []
    for _, row in df.iterrows():
        z_raw = int(row["Gaze Zone Number"])
        label_u = map_dmd_to_unified(z_raw)

        coords = row.iloc[3:].to_numpy(dtype=np.float32)
        L = (len(coords) // 3) * 3
        coords = coords[:L].reshape(-1, 3)
        coords_norm = normalize_coords(coords)
        n = coords_norm.shape[0]
        ei = get_fixed_edges(n)
        ea = build_edge_attr(coords_norm, ei.numpy()) if USE_EDGE_ATTR else None

        data = Data(
            x=torch.from_numpy(coords_norm),
            edge_index=ei.long(),
            edge_attr=ea,
            y=torch.tensor([label_u], dtype=torch.long)
        )
        graphs.append(data)
    return graphs

# -----------------------------
# Train / Eval
# -----------------------------
def train_one_epoch(model, loader, optim, crit, device):
    model.train()
    total = 0.0
    for data in loader:
        data = data.to(device)
        optim.zero_grad(set_to_none=True)
        out = model(data)
        loss = crit(out, data.y)
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
        preds.append(out.argmax(dim=1).cpu().numpy())
        labels.append(data.y.cpu().numpy())
    if not labels:
        return 0.0, 0.0, 0.0, np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    y_true = np.concatenate(labels)
    y_pred = np.concatenate(preds)
    micro_acc  = accuracy_score(y_true, y_pred)
    macro_acc  = balanced_accuracy_score(y_true, y_pred)
    macro_f1   = f1_score(y_true, y_pred, average='macro', zero_division=0)
    cm         = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    return micro_acc, macro_acc, macro_f1, cm

def print_metrics(tag, micro, macro, f1):
    print(f"[{tag}] micro-acc={micro:.4f} | macro-acc={macro:.4f} | macro-F1={f1:.4f}")

# -----------------------------
# Helpers
# -----------------------------
def make_model():
    return TransformerNet(num_node_features=3, output_dim=NUM_CLASSES).to(DEVICE)

def make_opt_sched(m):
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=STEP_SIZE, gamma=GAMMA)
    return opt, sch

# -----------------------------
# Main flow
# -----------------------------
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    # ==== A) Train on ALL DMD -> Test on DGW VAL ====
    print("\n==== [A] Train on ALL DMD -> Test on DGW (flipped) VAL ====")
    # Build ALL DMD graphs (RGB only; add IR similarly if needed)
    all_dmd_graphs = []
    for s in range(1, NUM_SUBJECTS + 1):
        csv_rgb = os.path.join(DMD_DIR, str(s), "Landmark_RGB_Valid.csv")
        if not os.path.isfile(csv_rgb):
            raise FileNotFoundError(csv_rgb)
        all_dmd_graphs.extend(build_graphs_from_dmd_csv(csv_rgb))
    train_loader_dmd = DataLoader(all_dmd_graphs, batch_size=BATCH_SIZE, shuffle=True,
                                  pin_memory=(DEVICE.type == "cuda"))

    # DGW val graphs
    dgw_val_graphs = build_graphs_from_dgw_csv(DGW_VAL_CSV)
    val_loader_dgw = DataLoader(dgw_val_graphs, batch_size=BATCH_SIZE, shuffle=False,
                                pin_memory=(DEVICE.type == "cuda"))

    modelA = make_model()
    optA, schA = make_opt_sched(modelA)
    crit = nn.CrossEntropyLoss()

    if DEVICE.type == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    for ep in range(1, EPOCHS + 1):
        loss = train_one_epoch(modelA, train_loader_dmd, optA, crit, DEVICE)
        schA.step()
        print(f"[A][epoch {ep:03d}] loss={loss:.4f}")
    if DEVICE.type == "cuda": torch.cuda.synchronize()
    train_time_A = time.perf_counter() - t0

    if DEVICE.type == "cuda": torch.cuda.synchronize()
    t1 = time.perf_counter()
    micro, macro, f1, cm = evaluate(modelA, val_loader_dgw, DEVICE)
    if DEVICE.type == "cuda": torch.cuda.synchronize()
    test_time_A = time.perf_counter() - t1

    print_metrics("A: DMD->DGW(val)", micro, macro, f1)
    print("Confusion Matrix (A):\n", cm)
    print(f"[A] Training time: {train_time_A:.2f}s | Test time: {test_time_A:.2f}s")

    # ==== B) Train on DGW TRAIN -> Test on each DMD subject; average ====
    print("\n==== [B] Train on DGW (flipped) TRAIN -> Test on each DMD subject ====")
    dgw_train_graphs = build_graphs_from_dgw_csv(DGW_TRAIN_CSV)
    train_loader_dgw = DataLoader(dgw_train_graphs, batch_size=BATCH_SIZE, shuffle=True,
                                  pin_memory=(DEVICE.type == "cuda"))

    modelB = make_model()
    optB, schB = make_opt_sched(modelB)

    if DEVICE.type == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    for ep in range(1, EPOCHS + 1):
        loss = train_one_epoch(modelB, train_loader_dgw, optB, crit, DEVICE)
        schB.step()
        print(f"[B][epoch {ep:03d}] loss={loss:.4f}")
    if DEVICE.type == "cuda": torch.cuda.synchronize()
    train_time_B = time.perf_counter() - t0
    print(f"[B] Training time: {train_time_B:.2f}s")

    # Per-subject testing
    accs_micro, accs_macro, f1s, cms = [], [], [], []
    total_test_time = 0.0

    for s in range(1, NUM_SUBJECTS + 1):
        csv_rgb = os.path.join(DMD_DIR, str(s), "Landmark_RGB_Valid.csv")
        test_graphs = build_graphs_from_dmd_csv(csv_rgb)
        test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE, shuffle=False,
                                 pin_memory=(DEVICE.type == "cuda"))

        if DEVICE.type == "cuda": torch.cuda.synchronize()
        t2 = time.perf_counter()
        micro_s, macro_s, f1_s, cm_s = evaluate(modelB, test_loader, DEVICE)
        if DEVICE.type == "cuda": torch.cuda.synchronize()
        dt = time.perf_counter() - t2
        total_test_time += dt

        print_metrics(f"B: DGW->DMD subject {s}", micro_s, macro_s, f1_s)
        print(f"[B] Subject {s} test time: {dt:.2f}s")
        print(f"[B] Confusion Matrix (subject {s}):\n{cm_s}")

        accs_micro.append(micro_s)
        accs_macro.append(macro_s)
        f1s.append(f1_s)
        cms.append(cm_s)

    # Averages
    avg_micro = float(np.mean(accs_micro)) if accs_micro else 0.0
    avg_macro = float(np.mean(accs_macro)) if accs_macro else 0.0
    avg_f1    = float(np.mean(f1s))        if f1s else 0.0
    avg_cm    = np.mean(np.stack(cms, axis=0), axis=0).round().astype(int) if cms else np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)

    print("\n==== [B] Averages over 15 DMD subjects ====")
    print_metrics("B: DGW->DMD (avg)", avg_micro, avg_macro, avg_f1)
    print("Average Confusion Matrix (rounded):\n", avg_cm)
    print(f"[B] Total test time across subjects: {total_test_time:.2f}s")
