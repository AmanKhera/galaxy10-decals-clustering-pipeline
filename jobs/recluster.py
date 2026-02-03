# jobs/recluster_job.py
# Creates scaler.pkl, pca20.pkl, hdbscan.pkl
# scaler.pkl, Standardizes 48-d embeddings, and makes PCA
# pca20.pkl, compresses 48-d to 20-d, making clustering more stable and faster
# hdbscan.pkl, gives cluster_labels (True labels from galaxy10_decals dataset) and membership probabilities
# galaxy10_clustered.csv, gives csv containing true_label, and which cluster it was assigned to, as well as membership_probability

import os
import numpy as np
import joblib
import hdbscan
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

EMB_PATH = "artifacts/embeddings/galaxy10_embeddings.npy"
Y_PATH   = "artifacts/embeddings/galaxy10_labels.npy"
OUT_MODELS = "artifacts/models"
OUT_RESULTS = "artifacts/results"

os.makedirs(OUT_MODELS, exist_ok=True)
os.makedirs(OUT_RESULTS, exist_ok=True)

def main():
    emb = np.load(EMB_PATH).astype(np.float32)
    y   = np.load(Y_PATH).astype(np.int64)

    scaler = StandardScaler()
    Z = scaler.fit_transform(emb)

    pca = PCA(n_components=20, random_state=42)
    Zp = pca.fit_transform(Z)

    joblib.dump(scaler, f"{OUT_MODELS}/scaler.pkl")
    joblib.dump(pca, f"{OUT_MODELS}/pca20.pkl")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=60,
        min_samples=10,
        metric="euclidean",
        prediction_data=True,
    )
    cluster_labels = clusterer.fit_predict(Zp)
    probs = clusterer.probabilities_

    joblib.dump(clusterer, f"{OUT_MODELS}/hdbscan.pkl")

    #save clustering results for galaxy10
    df = pd.DataFrame({
        "row_id": np.arange(len(cluster_labels), dtype=np.int32),
        "source": ["galaxy10"] * len(cluster_labels),
        "true_label": y,
        "cluster_id": cluster_labels.astype(np.int32),
        "membership_prob": probs.astype(np.float32),
        "outlier_score": (1.0 - probs.astype(np.float32)),
        "path": ["<Galaxy10_DECals.h5>"] * len(cluster_labels),
    })
    df.to_csv(f"{OUT_RESULTS}/galaxy10_clustered.csv", index=False)

    print("Saved:")
    print(" - artifacts/models/scaler.pkl")
    print(" - artifacts/models/pca20.pkl")
    print(" - artifacts/models/hdbscan.pkl")
    print(" - artifacts/results/galaxy10_clustered.csv", df.shape)

if __name__ == "__main__":
    main()
