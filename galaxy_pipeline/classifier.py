# galaxy_pipeline/classifier.py
import numpy as np
import tensorflow as tf

LABEL_NAMES = [
    "Disturbed", "Merging", "Round Smooth", "In-between Round Smooth", "Cigar",
    "Barred Spiral", "Tight Spiral", "Loose Spiral", "Edge-on (no bulge)", "Edge-on (with bulge)"
]

def predict_probs(model: tf.keras.Model, batch: tf.Tensor) -> np.ndarray:
    probs = model(batch, training=False).numpy()  # (1,10)
    return probs[0]

def top1(probs: np.ndarray) -> dict:
    idx = int(np.argmax(probs))
    return {"class_id": idx, "class_name": LABEL_NAMES[idx], "confidence": float(probs[idx])}
