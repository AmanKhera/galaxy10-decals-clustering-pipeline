# galaxy_pipeline/model_loader.py
import tensorflow as tf
from . import layers  #Call custom layers

def load_classifier(model_path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(model_path, compile=False)

def build_encoder(clf: tf.keras.Model, embedding_layer_index: int = -2) -> tf.keras.Model:
    """
    Default: layer right before final Dense softmax.
    """
    embed_output = clf.layers[embedding_layer_index].output
    return tf.keras.Model(inputs=clf.input, outputs=embed_output)
