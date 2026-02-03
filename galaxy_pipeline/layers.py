# galaxy_pipeline/layers.py
import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras.applications.efficientnet import preprocess_input

@K.utils.register_keras_serializable()
class CastToFloat16(K.layers.Layer):
    def call(self, x):
        x = tf.cast(x, tf.float16)
        return preprocess_input(x)

    def compute_output_shape(self, input_shape):
        return input_shape
