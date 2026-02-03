# galaxy_pipeline/knn_assign.py
# Given a new input this returns assigned cluster id, confidence, anomaly boolean + scores, readable cluster name

import numpy as np
import pandas as pd
import joblib

class KnnAssigner:
    def __init__(
        self,
        models_dir: str = "artifacts/models",
        emb_dir: str = "artifacts/embeddings",
        cluster_map_csv: str = "artifacts/results/cluster_map.csv",
    ):
        self.knn = joblib.load(f"{models_dir}/knn.pkl")
        self.cluster_ref = np.load(f"{emb_dir}/cluster_id_ref.npy").astype(np.int32)
        self.thr = float(np.load(f"{models_dir}/knn_dist_threshold.npy")[0])

        self.centroids = np.load(f"{models_dir}/cluster_centroids.npy", allow_pickle=True).item()
        self.dist_dists = np.load(f"{models_dir}/cluster_centroid_dists.npy", allow_pickle=True).item()

        self.cluster_map = pd.read_csv(cluster_map_csv).set_index("cluster_id")

    def assign(self, zp_new: np.ndarray, vote_min: float = 0.6) -> dict:
        zp_new = np.asarray(zp_new, dtype=np.float32).reshape(1, -1)
        dists, idxs = self.knn.kneighbors(zp_new)

        neigh_clusters = self.cluster_ref[idxs[0]]
        mean_dist = float(dists[0].mean())

        valid = neigh_clusters[neigh_clusters != -1]
        if len(valid) == 0:
            return {
                "assigned_cluster_id": -1,
                "cluster_name": "Noise/Unknown",
                "vote_conf": 0.0,
                "mean_knn_dist": mean_dist,
                "dist_to_center": float("nan"),
                "center_centrality": float("nan"),
                "is_anomaly": True,
            }

        vals, counts = np.unique(valid, return_counts=True)
        best = int(vals[np.argmax(counts)])
        vote_conf = float(np.max(counts) / len(valid))

        # distance to centroid + centrality percentile
        if best in self.centroids:
            c = self.centroids[best]
            dist_to_center = float(np.linalg.norm(zp_new.reshape(-1) - c))
            ref_d = self.dist_dists[best]
            centrality = float((ref_d > dist_to_center).mean())  # 0..1
        else:
            dist_to_center = float("nan")
            centrality = float("nan")

        is_anom = (mean_dist > self.thr) or (vote_conf < vote_min)

        if best in self.cluster_map.index:
            name = str(self.cluster_map.loc[best, "dominant_name"])
        else:
            name = f"Cluster {best}"

        return {
            "assigned_cluster_id": best,
            "cluster_name": name,
            "vote_conf": vote_conf,
            "mean_knn_dist": mean_dist,
            "dist_to_center": dist_to_center,
            "center_centrality": centrality,
            "is_anomaly": bool(is_anom),
        }
