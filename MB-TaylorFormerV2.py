import os
import sys
import csv
import time
import yaml
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mediapipe as mp

# =========================
# EDIT THESE PATHS IF NEEDED
# =========================
# Your DGW image folders
VAL_DIR   = r'D:\DGW\DGW_Dataset\cleaned_data\val_cleaned\val'
TRAIN_DIR = r'D:\DGW\DGW_Dataset\train\train'

# Output CSVs/logs
OUT_BASE = r'...\DGW_Dataset\Deblurred'
VAL_CSV_PATH   = os.path.join(OUT_BASE, 'val_deblurred_landmarks.csv')
TRAIN_CSV_PATH = os.path.join(OUT_BASE, 'train_deblurred_landmarks.csv')
VAL_LOG_PATH   = os.path.join(OUT_BASE, 'val_deblurred_failed.txt')
TRAIN_LOG_PATH = os.path.join(OUT_BASE, 'train_deblurred_failed.txt')

# MB-TaylorFormerV2 repo + files
REPO_ROOT    = r"...\MB-TaylorFormerV2-main\MB-TaylorFormerV2-main"
YAML_PATH    = r"...\MB-TaylorFormerV2-main\MB-TaylorFormerV2-main\Deblurring\Options\MB-TaylorFormerV2-XL.yml"
WEIGHTS_PATH = r"...\MB-TaylorFormerV2-main\MB-TaylorFormerV2-main\Deblurring\gopro-XL.pth"

# Model expects sizes multiple of:
MULT_FACTOR  = 8

# =========================
# Import model directly from file
# =========================
ARCHS_DIR = os.path.join(REPO_ROOT, "basicsr", "models", "archs")
if ARCHS_DIR not in sys.path:
    sys.path.insert(0, ARCHS_DIR)

from MB_TaylorFormerV2 import MB_TaylorFormer  # will raise if not found

def _load_yaml(path):
    with open(path, "r") as f:
        try:
            from yaml import CLoader as Loader
        except ImportError:
            from yaml import Loader
        return yaml.load(f, Loader=Loader)

class MBTaylorFormerDeblurrer:
    """FP32, reflect padding to multiple-of-8, returns BGR uint8."""
    def __init__(self, yaml_path: str, weights_path: str):
        if not os.path.isfile(yaml_path):
            raise FileNotFoundError(f"YAML not found: {yaml_path}")
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"Weights not found: {weights_path}")

        cfg = _load_yaml(yaml_path)
        if "network_g" not in cfg:
            raise RuntimeError("YAML missing 'network_g'")
        cfg["network_g"].pop("type", None)  # match repo test.py

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model = MB_TaylorFormer(**cfg["network_g"])

        ckpt = torch.load(weights_path, map_location="cpu")
        state = ckpt.get("params", ckpt)
        model.load_state_dict(state, strict=False)

        self.model = model.to(self.device)
        if self.device == "cuda":
            self.model = nn.DataParallel(self.model)
        self.model.eval()

    @torch.no_grad()
    def deblur_bgr(self, bgr_img: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)  # [1,3,H,W] FP32

        h, w = x.shape[2], x.shape[3]
        H = ((h + MULT_FACTOR) // MULT_FACTOR) * MULT_FACTOR
        W = ((w + MULT_FACTOR) // MULT_FACTOR) * MULT_FACTOR
        padh = H - h if h % MULT_FACTOR != 0 else 0
        padw = W - w if w % MULT_FACTOR != 0 else 0
        if padh > 0 or padw > 0:
            x = F.pad(x, (0, padw, 0, padh), mode="reflect")

        y = self.model(x)                              # [1,3,H',W'] in 0..1
        y = y[:, :, :h, :w]                            # unpad
        y = torch.clamp(y, 0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
        out_bgr = cv2.cvtColor((y * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
        return out_bgr

# =========================
# MediaPipe FaceMesh
# =========================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# =========================
# CSV header (interleaved triplets)
# =========================
def make_landmark_header():
    cols = ['filename']
    for i in range(478):
        cols += [f'x{i}', f'y{i}', f'z{i}']
    cols += ['Label']
    return cols

# =========================
# Process a split
# =========================
def process_images_split(directory, csv_file, log_file, deblurrer,
                         first_timing_bucket):
    """
    first_timing_bucket: dict with keys {'set', 'deblur_ms', 'landmarks_ms', 'done'}
    We record the very first successful sample timing across both splits.
    """
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)

    with open(csv_file, mode='w', newline='') as csv_f, open(log_file, mode='w') as log_f:
        writer = csv.writer(csv_f)
        writer.writerow(make_landmark_header())

        # loop class folders 1..9
        for subfolder in range(1, 10):
            folder_path = os.path.join(directory, str(subfolder))
            if not os.path.isdir(folder_path):
                continue

            for img_name in os.listdir(folder_path):
                if not img_name.lower().endswith('.png'):
                    continue

                img_path = os.path.join(folder_path, img_name)
                image = cv2.imread(img_path)
                if image is None:
                    log_f.write(f"{img_name} (read_failed)\n")
                    continue

                # 1) Deblur with timing
                t0 = time.perf_counter()
                try:
                    image_deblur = deblurrer.deblur_bgr(image)
                except Exception as e:
                    log_f.write(f"{img_name} (deblur_error: {e})\n")
                    continue
                t1 = time.perf_counter()
                deblur_ms = (t1 - t0) * 1000.0

                # 2) Landmark extraction with timing
                rgb_image = cv2.cvtColor(image_deblur, cv2.COLOR_BGR2RGB)
                t2 = time.perf_counter()
                result = face_mesh.process(rgb_image)
                t3 = time.perf_counter()
                landmarks_ms = (t3 - t2) * 1000.0

                if result.multi_face_landmarks:
                    landmarks = result.multi_face_landmarks[0].landmark
                    if len(landmarks) != 478:
                        log_f.write(f"{img_name} (unexpected_landmark_count={len(landmarks)})\n")
                        continue

                    # Record first-sample timing once
                    if not first_timing_bucket['done']:
                        first_timing_bucket['deblur_ms'] = deblur_ms
                        first_timing_bucket['landmarks_ms'] = landmarks_ms
                        first_timing_bucket['set'] = directory
                        first_timing_bucket['done'] = True

                    # write row (interleaved x,y,z)
                    row = [img_name]
                    for lm in landmarks:
                        row.extend([lm.x, lm.y, lm.z])
                    row.append(subfolder)
                    writer.writerow(row)
                else:
                    log_f.write(f"{img_name} (no_landmarks)\n")

# =========================
# Main
# =========================
def main():
    start_time_all = time.time()

    # Build deblurrer
    deblurrer = MBTaylorFormerDeblurrer(YAML_PATH, WEIGHTS_PATH)
    print("Deblur model loaded.")

    # bucket to store first-sample times across both splits
    first_timing = {'set': None, 'deblur_ms': None, 'landmarks_ms': None, 'done': False}

    # Process val then train (or swap order if you prefer)
    process_images_split(VAL_DIR,   VAL_CSV_PATH,   VAL_LOG_PATH,   deblurrer, first_timing)
    process_images_split(TRAIN_DIR, TRAIN_CSV_PATH, TRAIN_LOG_PATH, deblurrer, first_timing)

    face_mesh.close()

    end_time_all = time.time()
    print("Landmark extraction completed.")
    print(f'Total wall time: {end_time_all - start_time_all:.2f} seconds')

    if first_timing['done']:
        print(f"First sample timings (from set: {first_timing['set']}):")
        print(f"  Deblurring time: {first_timing['deblur_ms']:.2f} ms")
        print(f"  Landmark extraction time: {first_timing['landmarks_ms']:.2f} ms")
    else:
        print("Warning: no successful samples to measure first-sample timings.")

if __name__ == "__main__":
    main()
