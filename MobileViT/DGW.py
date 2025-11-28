import os, time, math, copy, gc
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets

# pip install timm thop scikit-learn
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

# ===================== CONFIG =====================
TRAIN_DIR = r'...\DGW\DGW_Dataset\train\train'
VAL_DIR   = r'...\DGW\DGW_Dataset\val\val'

MODEL_NAME     = 'mobilevit_s'   # MobileViT-Small in timm
EPOCHS         = 50
BATCH_SIZE     = 32
LR             = 3e-4
WEIGHT_DECAY   = 1e-4
NUM_WORKERS    = max(2, os.cpu_count() // 2)
AMP            = True            # mixed precision if CUDA
PIN_MEMORY     = True
PRINT_EVERY    = 50
SEED           = 1337


CB_BETA          = 0.995      # 0.99–0.999 usually
FOCAL_GAMMA      = 1.5        # 1.0–2.0
LABEL_SMOOTHING  = 0.05       # 0 to disable
EPS              = 1e-8

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

# ===================== UTILS ======================
def set_seed(seed=SEED):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)

def count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def try_compute_gflops(model, input_size_hw):
    """
    Returns GFLOPs estimate using THOP, WITHOUT mutating the original model/device.
    THOP reports MACs; FLOPs ≈ 2*MACs.
    """
    try:
        from thop import profile
    except Exception as e:
        print(f"[GFLOPs] Skipping (thop not available). Detail: {e}")
        return None

    # Work on a CPU copy to avoid device/type mismatches
    m = copy.deepcopy(model).eval().to('cpu')
    h, w = input_size_hw
    dummy = torch.randn(1, 3, h, w, device='cpu')
    try:
        macs, _ = profile(m, inputs=(dummy,), verbose=False)
        gflops = 2.0 * macs / 1e9
        return gflops
    except Exception as e:
        print(f"[GFLOPs] Skipping (profile failed). Detail: {e}")
        return None
    finally:
        del m, dummy
        gc.collect()

@torch.no_grad()
def measure_infer_time_and_mem(model, loader, device, warmup=3, steps=30, use_amp=True):
    model.eval()
    total_s, total_n = 0.0, 0
    peak_per_sample = 0.0
    steps_done = 0

    it = iter(loader)

    # Warmup
    for _ in range(min(warmup, len(loader))):
        try:
            images, _ = next(it)
        except StopIteration:
            break
        images = images.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=(use_amp and device.type == 'cuda')):
            _ = model(images)

    # Measure
    it = iter(loader)
    for batch in it:
        if steps_done >= steps:
            break
        images, _ = batch
        n = images.size(0)
        images = images.to(device, non_blocking=True)

        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            base_alloc = torch.cuda.memory_allocated(device)

        t0 = time.perf_counter()
        with torch.cuda.amp.autocast(enabled=(use_amp and device.type == 'cuda')):
            _ = model(images)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        if device.type == 'cuda':
            peak = torch.cuda.max_memory_allocated(device)
            extra = max(0, peak - base_alloc)
            per_sample = extra / n
            peak_per_sample = max(peak_per_sample, per_sample)

        total_s += (t1 - t0)
        total_n += n
        steps_done += 1

    ms_per_sample = (total_s / max(1, total_n)) * 1000.0
    mem_mb_sample = (peak_per_sample / (1024**2)) if device.type == 'cuda' else None
    return ms_per_sample, mem_mb_sample

@torch.no_grad()
def evaluate(model, loader, device, use_amp=True):
    model.eval()
    y_true, y_pred = [], []

    for images, targets in loader:
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=(use_amp and device.type == 'cuda')):
            logits = model(images)
        pred = logits.argmax(dim=1)
        y_true.extend(targets.cpu().tolist())
        y_pred.extend(pred.cpu().tolist())

    import numpy as np
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    micro_acc = accuracy_score(y_true, y_pred)          # overall accuracy
    macro_acc = balanced_accuracy_score(y_true, y_pred) # mean per-class recall
    macro_f1  = f1_score(y_true, y_pred, average='macro')
    return micro_acc, macro_acc, macro_f1

# ===================== LOSS: CB-Focal ======================
def make_cb_focal_criterion(class_counts, num_classes, beta=CB_BETA, gamma=FOCAL_GAMMA,
                            label_smoothing=LABEL_SMOOTHING, eps=EPS, device=None):
    """
    Class-Balanced Focal Loss with optional label smoothing.
    - class_counts: iterable length C (per-class sample counts from training set)
    """
    counts = torch.as_tensor(class_counts, dtype=torch.float32)
    if device is not None:
        counts = counts.to(device)

    # Class-Balanced weights via "Effective Number of Samples"
    if beta is not None:
        effective_num = 1.0 - torch.pow(torch.as_tensor(beta), counts)
        weights = (1.0 - beta) / (effective_num + eps)
    else:
        weights = torch.ones_like(counts)

    # Normalize to mean 1 (keeps loss scale stable)
    weights = weights * (num_classes / (weights.sum() + eps))

    def criterion(logits, targets):
        # logits: [B, C], targets: [B] (long)
        probs = torch.softmax(logits, dim=1).clamp_min(eps)  # [B, C]

        with torch.no_grad():
            y_true = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1.0)
            if label_smoothing and label_smoothing > 0:
                y_true = (1.0 - label_smoothing) * y_true + (label_smoothing / num_classes)

        # p_t = sum_c y_true_c * p_c
        p_t = (probs * y_true).sum(dim=1).clamp_min(eps)     # [B]
        focal = (1.0 - p_t).pow(gamma)                       # [B]

        alpha = weights.to(logits.device)                    # [C]
        ce_per_class = -y_true * torch.log(probs) * alpha    # [B, C]
        ce = ce_per_class.sum(dim=1)                         # [B]

        loss = (focal * ce).mean()
        return loss

    return criterion

# ===================== DATA ======================
def make_loaders(train_dir, val_dir, model_name, batch_size, num_workers):
    # Create a headless model to fetch transforms consistent with pretraining
    tmp_model = timm.create_model(model_name, pretrained=True, num_classes=0)
    cfg = resolve_data_config({}, model=tmp_model)
    train_tfms = create_transform(**cfg, is_training=True)
    eval_tfms  = create_transform(**cfg, is_training=False)

    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_ds   = datasets.ImageFolder(val_dir,   transform=eval_tfms)

    num_classes = len(train_ds.classes)
    print("Classes:", train_ds.classes)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=PIN_MEMORY, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=PIN_MEMORY, drop_last=False
    )

    input_h, input_w = cfg['input_size'][1], cfg['input_size'][2]  # (3,H,W)

    # ---- Compute class counts from training set for CB-Focal ----
    import numpy as np
    targets = np.array(train_ds.targets)  # list of ints
    class_counts = np.bincount(targets, minlength=num_classes)

    return train_loader, val_loader, num_classes, (input_h, input_w), class_counts

# ===================== TRAIN =====================
def train():
    set_seed(SEED)
    print(f"Device: {DEVICE} | AMP: {AMP and DEVICE.type=='cuda'}")

    (train_loader, val_loader, num_classes, in_hw, class_counts) = make_loaders(
        TRAIN_DIR, VAL_DIR, MODEL_NAME, BATCH_SIZE, NUM_WORKERS
    )

    # Model
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=num_classes)
    model.to(DEVICE)

    # Optim & sched
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=(AMP and DEVICE.type == 'cuda'))

    # ---- Build CB-Focal criterion using train set class_counts ----
    criterion = make_cb_focal_criterion(
        class_counts=class_counts,
        num_classes=num_classes,
        beta=CB_BETA,
        gamma=FOCAL_GAMMA,
        label_smoothing=LABEL_SMOOTHING,
        device=DEVICE
    )

    # Stats (params + GFLOPs w/o moving the live model)
    total_params, trainable_params = count_params(model)
    print(f"Params: {total_params/1e6:.3f}M (trainable {trainable_params/1e6:.3f}M)")
    gflops = try_compute_gflops(model, in_hw)
    if gflops is not None:
        print(f"Approx GFLOPs (1x{in_hw[0]}x{in_hw[1]}): {gflops:.2f}")

    # Safety: ensure model is on right device before training
    p = next(model.parameters())
    assert (p.is_cuda == (DEVICE.type == 'cuda')), f"Model device mismatch: {p.device}"

    # Epoch loop
    best_macro_f1 = -1.0
    epoch_times = []

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        seen = 0

        for step, (images, targets) in enumerate(train_loader, 1):
            images  = images.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(AMP and DEVICE.type == 'cuda')):
                logits = model(images)
                loss = criterion(logits, targets)  # <-- CB-Focal here

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            seen += images.size(0)

            if step % PRINT_EVERY == 0:
                print(f"Epoch {epoch:03d} | step {step:04d}/{len(train_loader)} | loss {loss.item():.4f}")

        scheduler.step()
        epoch_time = time.time() - t0
        epoch_times.append(epoch_time)

        train_loss = running_loss / max(1, seen)
        micro_acc, macro_acc, macro_f1 = evaluate(model, val_loader, DEVICE, use_amp=AMP)

        print(f"[Epoch {epoch:03d}/{EPOCHS}] "
              f"train_loss={train_loss:.4f} | "
              f"val_micro_acc={micro_acc:.4f} | val_macro_acc={macro_acc:.4f} | val_macro_f1={macro_f1:.4f} | "
              f"epoch_time={epoch_time:.2f}s | lr={optimizer.param_groups[0]['lr']:.2e}")

        # Save best by Macro-F1
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            torch.save(model.state_dict(), "mobilevit_s_best.pth")

    # Final eval metrics
    final_micro, final_macro_acc, final_macro_f1 = evaluate(model, val_loader, DEVICE, use_amp=AMP)
    print("\n=== Final Validation Metrics ===")
    print(f"Micro Accuracy : {final_micro:.4f}")
    print(f"Macro Accuracy : {final_macro_acc:.4f}")
    print(f"Macro F1       : {final_macro_f1:.4f}")

    # Inference latency & memory (per sample)
    ms_ps, mem_mb_ps = measure_infer_time_and_mem(model, val_loader, DEVICE, warmup=3, steps=30, use_amp=AMP)
    mem_str = f"{mem_mb_ps:.2f} MB/sample" if mem_mb_ps is not None else "N/A (CPU)"
    print(f"Inference latency: {ms_ps:.2f} ms/sample | Peak extra memory: {mem_str}")

    # Report params & GFLOPs again
    print(f"Params: {total_params/1e6:.3f}M (trainable {trainable_params/1e6:.3f}M)")
    if gflops is not None:
        print(f"Approx GFLOPs: {gflops:.2f}")

    # Per-epoch training times
    print("\nPer-epoch training time (s):")
    for i, t in enumerate(epoch_times, 1):
        print(f"  Epoch {i:02d}: {t:.2f}s")

if __name__ == "__main__":
    train()
