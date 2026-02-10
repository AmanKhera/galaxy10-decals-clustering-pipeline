# galaxy_pipeline/infer.py
from __future__ import annotations

import os
import glob
import shutil
import csv
from datetime import datetime

import numpy as np

from .model_loader import load_classifier, build_encoder
from .transform import load_scaler_pca, to_pca_space
from .knn_assign import assign_with_knn  # adjust if your function name differs
from .embeddings import embed_images     # adjust if your function name differs


DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "/app/artifacts")

NEW_DROP_DIR = os.path.join(DATA_DIR, "new_drop")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed_images")
ANOM_IMG_DIR = os.path.join(DATA_DIR, "anomalies", "images")

MODELS_DIR = os.path.join(ARTIFACTS_DIR, "models")
RESULTS_DIR = os.path.join(ARTIFACTS_DIR, "results")

MODEL_PATH = os.path.join("/app/best_model", "galaxy_b3_final_BEST_100TP.keras")
OUT_CSV = os.path.join(RESULTS_DIR, "new_inference.csv")


def ensure_dirs():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(ANOM_IMG_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


def append_rows(csv_path: str, rows: list[dict]):
    fieldnames = list(rows[0].keys()) if rows else []
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerows(rows)


def main():
    ensure_dirs()

    paths = sorted(glob.glob(os.path.join(NEW_DROP_DIR, "*.jpg")))
    print(f"[infer] Found {len(paths)} images in {NEW_DROP_DIR}")

    if len(paths) == 0:
        print("[infer] Nothing to do. Exiting.")
        return

    # Load model + build encoder
    print(f"[infer] Loading model: {MODEL_PATH}")
    clf = load_classifier(MODEL_PATH)
    encoder = build_encoder(clf)

    # Load transforms + knn
    print(f"[infer] Loading scaler+pca from: {MODELS_DIR}")
    scaler, pca = load_scaler_pca(MODELS_DIR)

    print(f"[infer] Loading kNN from: {MODELS_DIR}/knn.pkl")
    # assign_with_knn should load knn internally OR accept a knn_path/models_dir
    # (we pass models_dir so you can load knn.pkl inside knn_assign.py)
    models_dir = MODELS_DIR

    # Embed
    print("[infer] Embedding images...")
    emb = embed_images(encoder, paths)   # expected shape (N, D)
    emb = np.asarray(emb, dtype=np.float32)
    print(f"[infer] Embeddings shape: {emb.shape}")

    # PCA space
    Zp = to_pca_space(emb, scaler, pca)
    print(f"[infer] PCA shape: {Zp.shape}")

    # Assign clusters + anomaly
    print("[infer] Assigning clusters via kNN...")
    assignments = assign_with_knn(Zp, paths, models_dir=models_dir)
    # expected: list[dict] with at least: path, cluster, is_anomaly, score/dist fields

    # Write CSV
    now = datetime.utcnow().isoformat()
    rows = []
    anom_count = 0

    for a in assignments:
        src = a.get("path") or a.get("image_path") or ""
        is_anom = bool(a.get("is_anomaly", False))
        if is_anom:
            anom_count += 1
            try:
                shutil.copy2(src, os.path.join(ANOM_IMG_DIR, os.path.basename(src)))
            except Exception as e:
                print(f"[warn] Failed to copy anomaly image {src}: {e}")

        row = {
            "timestamp_utc": now,
            "filename": os.path.basename(src),
            "path": src,
            "cluster": a.get("cluster", a.get("cluster_id", "")),
            "is_anomaly": is_anom,
            "score": a.get("score", a.get("distance", "")),
        }
        rows.append(row)

    if rows:
        append_rows(OUT_CSV, rows)
        print(f"[infer] Wrote {len(rows)} rows to {OUT_CSV}")

    # Move processed images
    moved = 0
    for p in paths:
        try:
            shutil.move(p, os.path.join(PROCESSED_DIR, os.path.basename(p)))
            moved += 1
        except Exception as e:
            print(f"[warn] Failed to move {p}: {e}")

    print(f"[infer] Done. processed={moved}, anomalies={anom_count}")
    print(f"[infer] anomalies copied to: {ANOM_IMG_DIR}")
    print(f"[infer] processed moved to: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
