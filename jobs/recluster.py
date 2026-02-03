# jobs/recluster_job.py
# Creates scaler.pkl, pca20.pkl, hdbscan.pkl
# scaler.pkl, Standardizes 48-d embeddings, and makes PCA
# pca20.pkl, compresses 48-d to 20-d, making clustering more stable and faster
# hdbscan.pkl, gives cluster_labels (True labels from galaxy10_decals dataset) and membership probabilities
# galaxy10_clustered.csv, gives csv containing true_label, and which cluster it was assigned to, as well as membership_probability

import os
import numpy as np
import pandas as pd
import joblib
import hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from galaxy_pipeline.cluster_map import build_cluster_map

EMB_PATH = "artifacts/embeddings/galaxy10_embeddings.npy"
Y_PATH   = "artifacts/embeddings/galaxy10_labels.npy"

OUT_MODELS  = "artifacts/models"
OUT_RESULTS = "artifacts/results"
os.makedirs(OUT_MODELS, exist_ok=True)
os.makedirs(OUT_RESULTS, exist_ok=True)

def main():
    # ---------- A1) load reference embeddings ----------
    emb = np.load(EMB_PATH).astype(np.float32)
    y   = np.load(Y_PATH).astype(np.int64)

    # ---------- A2) fit scaler + PCA ----------
    scaler = StandardScaler()
    Z = scaler.fit_transform(emb)

    pca = PCA(n_components=20, random_state=42)
    Zp = pca.fit_transform(Z)

    joblib.dump(scaler, f"{OUT_MODELS}/scaler.pkl")
    joblib.dump(pca,    f"{OUT_MODELS}/pca20.pkl")

    # ---------- A3) fit HDBSCAN ----------
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=60,
        min_samples=10,
        metric="euclidean",
        prediction_data=True,
    )
    cluster_labels = clusterer.fit_predict(Zp)
    probs = clusterer.probabilities_.astype(np.float32)

    # Optional but useful
    joblib.dump(clusterer, f"{OUT_MODELS}/hdbscan.pkl")

    # Optional but VERY useful for online kNN assignment later
    np.save(f"{OUT_MODELS}/zp_ref.npy", Zp.astype(np.float32))
    np.save(f"{OUT_MODELS}/cluster_labels_ref.npy", cluster_labels.astype(np.int32))

    # ---------- A4) write galaxy10_clustered.csv ----------
    clustered_csv = f"{OUT_RESULTS}/galaxy10_clustered.csv"
    df = pd.DataFrame({
        "row_id": np.arange(len(cluster_labels), dtype=np.int32),
        "source": ["galaxy10"] * len(cluster_labels),
        "true_label": y,
        "cluster_id": cluster_labels.astype(np.int32),
        "membership_prob": probs,
        "outlier_score": (1.0 - probs),
        "path": ["<Galaxy10_DECals.h5>"] * len(cluster_labels),
    })
    df.to_csv(clustered_csv, index=False)

    # ---------- B) build cluster_map.csv from the fresh clustered CSV ----------
    cluster_map_csv = f"{OUT_RESULTS}/cluster_map.csv"
    build_cluster_map(clustered_csv, cluster_map_csv)

    print("Saved:")
    print(" - artifacts/models/scaler.pkl")
    print(" - artifacts/models/pca20.pkl")
    print(" - artifacts/models/hdbscan.pkl")
    print(" - artifacts/models/zp_ref.npy")
    print(" - artifacts/models/cluster_labels_ref.npy")
    print(" - artifacts/results/galaxy10_clustered.csv")
    print(" - artifacts/results/cluster_map.csv")

if __name__ == "__main__":
    main()

