# galaxy_pipeline/infer.py
# Actually what is called to infer new images

import shutil
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from .model_loader import load_classifier, build_encoder
from jobs.transform import load_scaler_pca, to_pca_space
from .knn_assign import KnnAssigner
from .preprocess import load_image_rgb, make_batch

class GalaxyInferencePipeline:
    def __init__(
        self,
        model_path: str,
        models_dir: str = "artifacts/models",
        incoming_dir: str = "data/incoming/images",
        anomaly_dir: str = "data/anomalies/images",
        log_csv: str = "artifacts/results/new_inference.csv",
        embedding_layer_index: int = -2,
    ):
        self.incoming_dir = Path(incoming_dir)
        self.anomaly_dir = Path(anomaly_dir)
        self.log_csv = Path(log_csv)
        self.incoming_dir.mkdir(parents=True, exist_ok=True)
        self.anomaly_dir.mkdir(parents=True, exist_ok=True)
        self.log_csv.parent.mkdir(parents=True, exist_ok=True)

        # Load once
        self.clf = load_classifier(model_path)
        self.encoder = build_encoder(self.clf, embedding_layer_index=embedding_layer_index)

        self.scaler, self.pca = load_scaler_pca(models_dir)
        self.assigner = KnnAssigner(models_dir=models_dir)

    def infer_one(self, image_path: str, vote_min: float = 0.6) -> dict:
        image_path = Path(image_path)

        # 1) preprocess -> (1,256,256,3)
        img = load_image_rgb(str(image_path))
        batch = make_batch(img)

        # 2) embedding -> (1,48)
        e = self.encoder(batch, training=False).numpy().astype(np.float32)

        # 3) transform to PCA space -> (20,)
        zp = to_pca_space(e, self.scaler, self.pca).reshape(-1)

        # 4) kNN assign + anomaly decision
        out = self.assigner.assign(zp, vote_min=vote_min)

        # 5) route file
        dest_dir = self.anomaly_dir if out["is_anomaly"] else self.incoming_dir
        dest_path = dest_dir / image_path.name
        if not dest_path.exists():
            shutil.copy2(image_path, dest_path)

        # 6) log row
        row = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "source_path": str(image_path),
            "stored_path": str(dest_path),
            **out,
        }

        df = pd.DataFrame([row])
        if self.log_csv.exists():
            df.to_csv(self.log_csv, mode="a", header=False, index=False)
        else:
            df.to_csv(self.log_csv, index=False)

        return row
