# galaxy_pipeline/transform.py
# Allows new inputs to be assigned clusters via k-NN instead of re-running hdbscan

import numpy as np
import joblib

def load_scaler_pca(models_dir: str):
    scaler = joblib.load(f"{models_dir}/scaler.pkl")
    pca    = joblib.load(f"{models_dir}/pca20.pkl")
    return scaler, pca

def to_pca_space(emb: np.ndarray, scaler, pca) -> np.ndarray:
    """
    emb: (N, D) float32
    returns: (N, 20) float32
    """
    Z = scaler.transform(emb)
    Zp = pca.transform(Z)
    return Zp.astype(np.float32)
