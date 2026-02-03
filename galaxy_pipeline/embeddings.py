# galaxy_pipeline/embeddings.py
import os
import numpy as np
import pandas as pd
import h5py
import tensorflow as tf

from .model_loader import load_classifier, build_encoder
from .preprocess import preprocess_uint8_batch, preprocess_path

def _infer_target_size_from_model(model: tf.keras.Model) -> tuple[int, int]:
    shp = model.input_shape  #(None, H, W, 3)
    if shp[1] is None or shp[2] is None:
        raise ValueError(f"Model input_shape has unknown spatial dims: {shp}")
    return (int(shp[1]), int(shp[2]))

def build_encoder_from_path(model_path: str, embedding_layer_index: int = -2):
    clf = load_classifier(model_path)
    enc = build_encoder(clf, embedding_layer_index=embedding_layer_index)
    target_size = _infer_target_size_from_model(enc)
    return enc, target_size

def embed_galaxy10_h5(
    h5_path: str,
    model_path: str,
    out_dir: str,
    batch_size: int = 48,
    embedding_layer_index: int = -2,
    images_key: str = "images",
    labels_key: str = "ans",
):
    os.makedirs(out_dir, exist_ok=True)

    encoder, target_size = build_encoder_from_path(model_path, embedding_layer_index)

    with h5py.File(h5_path, "r") as f:
        X = f[images_key]
        y = f[labels_key]
        N = X.shape[0]

        # infer embedding dim
        d = int(encoder(tf.zeros((1, *target_size, 3), dtype=tf.float32), training=False).shape[-1])

        embeddings = np.zeros((N, d), dtype=np.float32)
        labels = np.zeros((N,), dtype=np.int64)

        for i in range(0, N, batch_size):
            xb = X[i:i+batch_size]
            yb = y[i:i+batch_size]

            xb = preprocess_uint8_batch(xb, target_size)  #(B,H,W,3) float32 0..255
            eb = encoder(xb, training=False).numpy().astype(np.float32)

            embeddings[i:i+len(eb)] = eb
            labels[i:i+len(yb)] = np.asarray(yb, dtype=np.int64)

    np.save(os.path.join(out_dir, "galaxy10_embeddings.npy"), embeddings)
    np.save(os.path.join(out_dir, "galaxy10_labels.npy"), labels)

    return embeddings.shape, labels.shape


def embed_new_from_manifest(
    manifest_csv: str,
    inference_csv: str,   #to skip already-processed stored_path
    model_path: str,
    out_dir: str,
    meta_out_csv: str,
    batch_size: int = 48,
    embedding_layer_index: int = -2,
):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(meta_out_csv), exist_ok=True)

    encoder, target_size = build_encoder_from_path(model_path, embedding_layer_index)

    m = pd.read_csv(manifest_csv)
    if "path" not in m.columns:
        raise ValueError(f"{manifest_csv} must contain a 'path' column. Found {list(m.columns)}")
    m["path"] = m["path"].astype(str)

    if "status" in m.columns:
        ok_mask = m["status"].astype(str).isin(["ok", "exists"])
        m = m[ok_mask].copy()

    manifest_paths = m["path"].dropna().unique().tolist()
    manifest_paths = [p for p in manifest_paths if os.path.exists(p)]

    processed_paths = set()
    if os.path.exists(inference_csv) and os.path.getsize(inference_csv) > 0:
        inf = pd.read_csv(inference_csv)
        if "stored_path" not in inf.columns:
            raise ValueError(f"{inference_csv} must contain 'stored_path'. Found {list(inf.columns)}")
        processed_paths = set(inf["stored_path"].astype(str).dropna().unique().tolist())

    processed_ids = {os.path.basename(p) for p in processed_paths}
    to_embed = [p for p in manifest_paths if os.path.basename(p) not in processed_ids]

    if len(to_embed) == 0:
        return (0, 0)

    # infer embedding dim
    d = int(encoder(tf.zeros((1, *target_size, 3), dtype=tf.float32), training=False).shape[-1])

    M = len(to_embed)
    emb = np.zeros((M, d), dtype=np.float32)

    # batched path -> tensor pipeline
    for i in range(0, M, batch_size):
        batch_paths = to_embed[i:i+batch_size]
        xb = tf.stack([preprocess_path(tf.constant(p), target_size) for p in batch_paths], axis=0)
        eb = encoder(xb, training=False).numpy().astype(np.float32)
        emb[i:i+len(eb)] = eb

    labels = np.full((M,), -1, dtype=np.int64)

    np.save(os.path.join(out_dir, "new_embeddings.npy"), emb)
    np.save(os.path.join(out_dir, "new_labels.npy"), labels)

    pd.DataFrame({"stored_path": to_embed}).to_csv(meta_out_csv, index=False)

    return emb.shape, labels.shape


def append_new_to_all(
    out_dir: str,
    meta_new_csv: str,
    galaxy10_emb_path: str,
    galaxy10_lab_path: str,
    all_emb_name: str = "embeddings_all.npy",
    all_lab_name: str = "labels_all.npy",
    all_meta_csv: str = "artifacts/results/embeddings_all_meta.csv",
):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(all_meta_csv), exist_ok=True)

    ALL_EMB  = os.path.join(out_dir, all_emb_name)
    ALL_LAB  = os.path.join(out_dir, all_lab_name)

    emb_g10 = np.load(galaxy10_emb_path).astype(np.float32)
    y_g10   = np.load(galaxy10_lab_path).astype(np.int64)
    N0 = emb_g10.shape[0]

    emb_new = np.load(os.path.join(out_dir, "new_embeddings.npy")).astype(np.float32)
    meta_new = pd.read_csv(meta_new_csv)
    if len(meta_new) != emb_new.shape[0]:
        raise ValueError("meta_new rows must match new_embeddings rows")

    new_ids = meta_new["stored_path"].astype(str).map(os.path.basename).tolist()

    if os.path.exists(ALL_EMB) and os.path.exists(ALL_LAB) and os.path.exists(all_meta_csv):
        emb_all = np.load(ALL_EMB).astype(np.float32)
        y_all   = np.load(ALL_LAB).astype(np.int64)
        meta_all = pd.read_csv(all_meta_csv)

        existing_new_ids = set(
            meta_all[meta_all["source"] == "new"]["path"].astype(str).map(os.path.basename).tolist()
        )
    else:
        emb_all = emb_g10
        y_all   = y_g10
        meta_all = pd.DataFrame({
            "row_id": np.arange(N0),
            "source": ["galaxy10"] * N0,
            "source_idx": list(range(N0)),
            "path": ["<Galaxy10_DECals.h5>"] * N0,
            "true_label": y_g10,
        })
        existing_new_ids = set()

    keep_mask = np.array([bid not in existing_new_ids for bid in new_ids], dtype=bool)
    k = int(keep_mask.sum())
    if k == 0:
        return emb_all.shape, meta_all.shape

    emb_to_add = emb_new[keep_mask]
    y_to_add   = np.full((k,), -1, dtype=np.int64)

    paths_to_add = meta_new["stored_path"].astype(str).tolist()
    paths_to_add = [p for p, keep in zip(paths_to_add, keep_mask) if keep]

    emb_all = np.concatenate([emb_all, emb_to_add], axis=0)
    y_all   = np.concatenate([y_all, y_to_add], axis=0)

    start = len(meta_all)
    new_meta_rows = pd.DataFrame({
        "row_id": np.arange(start, start + k),
        "source": ["new"] * k,
        "source_idx": list(range(k)),
        "path": paths_to_add,
        "true_label": y_to_add,
    })
    meta_all = pd.concat([meta_all, new_meta_rows], ignore_index=True)

    np.save(ALL_EMB, emb_all)
    np.save(ALL_LAB, y_all)
    meta_all.to_csv(all_meta_csv, index=False)

    return emb_all.shape, meta_all.shape
