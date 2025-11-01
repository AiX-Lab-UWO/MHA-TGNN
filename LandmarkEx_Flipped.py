import os
import sys
import csv
import glob
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

# ---- if mediapipe not installed, uncomment:
# pip install mediapipe==0.10.9 opencv-python
import mediapipe as mp

# -----------------------------
# Paths (edit if needed)
# -----------------------------
VAL_DIR_IN   = r"...\DGW_Dataset\val"
TRAIN_DIR_IN = r"...\DGW_Dataset\train"

VAL_DIR_OUT_IMG   = r"...\DGW\DGW_Dataset\flipped\val"
TRAIN_DIR_OUT_IMG = r"...\DGW\DGW_Dataset\flipped\train"

VAL_CSV_OUT   = r"...\DGW\DGW_Dataset\flipped\val_flipped_landmarks.csv"
TRAIN_CSV_OUT = r"...\DGW\DGW_Dataset\flipped\train_flipped_landmarks.csv"

# -----------------------------
# MediaPipe config
# -----------------------------
MAX_FACES = 1
STATIC_IMAGE_MODE = True
REFINE = True      # -> 478 landmarks
MIN_DET_CONF = 0.5
MIN_TRACK_CONF = 0.5

# -----------------------------
# Utils
# -----------------------------
def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def imread_rgb(path: str):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def imwrite_bgr(path: str, rgb: np.ndarray):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, bgr)

def list_images_in_class_dir(root: str):
    """Yield (img_path, class_id) for class folders 1..9."""
    all_pairs = []
    for c in range(1, 10):
        c_dir = os.path.join(root, str(c))
        if not os.path.isdir(c_dir):
            continue
        # common image extensions
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
            for p in glob.glob(os.path.join(c_dir, ext)):
                all_pairs.append((p, c))
    return sorted(all_pairs)

def flip_horizontal(rgb: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(rgb[:, ::-1, :])

def mediapipe_facemesh():
    mp_face = mp.solutions.face_mesh
    fm = mp_face.FaceMesh(
        static_image_mode=STATIC_IMAGE_MODE,
        max_num_faces=MAX_FACES,
        refine_landmarks=REFINE,
        min_detection_confidence=MIN_DET_CONF,
        min_tracking_confidence=MIN_TRACK_CONF
    )
    return fm

def extract_478_landmarks(face_landmarks, image_w: int, image_h: int):
    """
    Returns np.array [478, 3] of normalized (x,y,z) in image coordinates:
      x,y in [0..1] relative to width/height; z is MediaPipe relative depth.
    """
    lm = face_landmarks.landmark
    n = len(lm)
    # Expect 478 with refine_landmarks=True (468 + irises)
    coords = np.zeros((n, 3), dtype=np.float32)
    for i, p in enumerate(lm):
        coords[i, 0] = p.x  # normalized [0,1] across width
        coords[i, 1] = p.y  # normalized [0,1] across height
        coords[i, 2] = p.z  # relative depth (approximately in normalized image coords)
    return coords

def row_from_coords(filename: str, coords: np.ndarray, label_1_to_9: int):
    flat = coords.reshape(-1)  # 478*3
    row = [filename] + [float(v) for v in flat] + [int(label_1_to_9)]
    return row

def process_split(split_name: str,
                  in_root: str,
                  out_img_root: str,
                  out_csv_path: str,
                  save_flipped_images: bool = True):
    """
    Walk class folders 1..9, flip images, extract 478 landmarks.
    Writes CSV with schema: filename, x0,y0,z0,...,x477,y477,z477, Label
    """
    ensure_dir(os.path.dirname(out_csv_path))
    ensure_dir(out_img_root)

    images = list_images_in_class_dir(in_root)
    if not images:
        print(f"[{split_name}] No images found under: {in_root}")
        return

    mp_fm = mediapipe_facemesh()

    # CSV header
    n_points = 478 if REFINE else 468
    header = ["filename"]
    for i in range(n_points):
        header += [f"x{i}", f"y{i}", f"z{i}"]
    header += ["Label"]

    n_ok, n_fail = 0, 0
    t0 = time.time()
    with open(out_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for idx, (img_path, cls_id) in enumerate(images, 1):
            try:
                rgb = imread_rgb(img_path)
                if rgb is None:
                    n_fail += 1
                    print(f"[{split_name}] (skip) cannot read image: {img_path}")
                    continue

                # 1) flip
                rgb_flipped = flip_horizontal(rgb)

                # 2) save flipped (same relative class/filename)
                rel = os.path.relpath(img_path, in_root)
                out_img_path = os.path.join(out_img_root, rel)
                if save_flipped_images:
                    imwrite_bgr(out_img_path, rgb_flipped)

                # 3) landmarks
                h, w = rgb_flipped.shape[:2]
                results = mp_fm.process(rgb_flipped)
                if not results.multi_face_landmarks:
                    n_fail += 1
                    print(f"[{split_name}] (no face) {img_path}")
                    continue

                coords = extract_478_landmarks(results.multi_face_landmarks[0], w, h)

                # 4) CSV row (store the flipped filename path for traceability)
                row = row_from_coords(out_img_path, coords, cls_id)  # Label stays 1..9
                writer.writerow(row)

                n_ok += 1
                if idx % 200 == 0:
                    dt = time.time() - t0
                    print(f"[{split_name}] {idx}/{len(images)} processed | ok={n_ok} fail={n_fail} | {dt:.1f}s")

            except Exception as e:
                n_fail += 1
                print(f"[{split_name}] (error) {img_path}: {e}")
                traceback.print_exc()

    dt = time.time() - t0
    print(f"[{split_name}] DONE. Images={len(images)}  OK={n_ok}  FAIL={n_fail}  Time={dt:.1f}s")
    print(f"[{split_name}] CSV -> {out_csv_path}")
    if save_flipped_images:
        print(f"[{split_name}] Flipped images saved under -> {out_img_root}")

if __name__ == "__main__":
    # VAL split
    process_split(
        split_name="VAL",
        in_root=VAL_DIR_IN,
        out_img_root=VAL_DIR_OUT_IMG,
        out_csv_path=VAL_CSV_OUT,
        save_flipped_images=True
    )

    # TRAIN split
    process_split(
        split_name="TRAIN",
        in_root=TRAIN_DIR_IN,
        out_img_root=TRAIN_DIR_OUT_IMG,
        out_csv_path=TRAIN_CSV_OUT,
        save_flipped_images=True
    )

    print("All done. Next step: cross-dataset training with zone merging/reindexing.")
