# jobs/build_knn_artifacts.py
# Code is ran once after reclustering (running hdbscan) to create artifacts which will be used
# for the K-NN



import os
import numpy as np
import pandas as pd
import joblib
from sklearn.neighbors import NearestNeighbors

from galaxy_pipeline.transform import load_scaler_pca, to_pca_space

EMB_PATH = "artifacts/embeddings/galaxy10_embeddings.npy"
CLUSTERED_CSV = "artifacts/results/galaxy10_clustered.csv"

MODELS_DIR = "artifacts/models"
EMB_DIR    = "artifacts/embeddings"

K_NEIGHBORS = 15
THR_PERCENTILE = 99  # 97-99

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(EMB_DIR, exist_ok=True)

    emb = np.load(EMB_PATH).astype(np.float32)
    df  = pd.read_csv(CLUSTERED_CSV)
    cluster_ids = df["cluster_id"].values.astype(np.int32)

    scaler, pca = load_scaler_pca(MODELS_DIR)
    Zp = to_pca_space(emb, scaler, pca)  # (N,20)

    # Save reference PCA-space + cluster labels
    np.save(f"{EMB_DIR}/Zp_ref.npy", Zp)
    np.save(f"{EMB_DIR}/cluster_id_ref.npy", cluster_ids)

    # Fit + save kNN
    knn = NearestNeighbors(n_neighbors=K_NEIGHBORS, metric="euclidean")
    knn.fit(Zp)
    joblib.dump(knn, f"{MODELS_DIR}/knn.pkl")

    # Compute distance threshold for anomaly gating
    dists, _ = knn.kneighbors(Zp)
    mean_knn_dist = dists[:, 1:].mean(axis=1)              # skip self distance
    base = mean_knn_dist[cluster_ids != -1]                # non-noise only
    thr = float(np.percentile(base, THR_PERCENTILE))
    np.save(f"{MODELS_DIR}/knn_dist_threshold.npy", np.array([thr], dtype=np.float32))

    # Centroids + per-cluster distance distributions
    centroids = {}
    dist_dists = {}

    for cid in sorted(set(cluster_ids)):
        if cid == -1:
            continue
        pts = Zp[cluster_ids == cid]
        c = pts.mean(axis=0)
        centroids[int(cid)] = c.astype(np.float32)

        d = np.linalg.norm(pts - c, axis=1).astype(np.float32)
        dist_dists[int(cid)] = d

    np.save(f"{MODELS_DIR}/cluster_centroids.npy", centroids, allow_pickle=True)
    np.save(f"{MODELS_DIR}/cluster_centroid_dists.npy", dist_dists, allow_pickle=True)

    print("Saved:")
    print(" - artifacts/models/knn.pkl")
    print(" - artifacts/models/knn_dist_threshold.npy =", thr)
    print(" - artifacts/models/cluster_centroids.npy")
    print(" - artifacts/models/cluster_centroid_dists.npy")
    print(" - artifacts/embeddings/Zp_ref.npy", Zp.shape)
    print(" - artifacts/embeddings/cluster_id_ref.npy", cluster_ids.shape)

if __name__ == "__main__":
    main()
