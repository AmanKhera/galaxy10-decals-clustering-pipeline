# galaxy_pipeline/model_loader.py
import tensorflow as tf
from . import layers  #Call custom layers

def load_classifier(model_path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(model_path, compile=False)
