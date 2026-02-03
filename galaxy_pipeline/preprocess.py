# galaxy_pipeline/preprocess.py
import tensorflow as tf

def preprocess_uint8_batch(x_uint8: tf.Tensor, target_size: tuple[int, int]) -> tf.Tensor:
    """
    x_uint8: (B,H,W,3) uint8 or float
    returns: (B,target_h,target_w,3) float32 in 0..255
    """
    x = tf.cast(x_uint8, tf.float32)
    x = tf.image.resize(x, target_size, method="bilinear")
    x = tf.clip_by_value(x, 0.0, 255.0)
    return x

def preprocess_path(path: tf.Tensor, target_size: tuple[int, int]) -> tf.Tensor:
    """
    path: scalar tf.string
    returns: (target_h,target_w,3) float32 0..255
    """
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)  # uint8
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, target_size, method="bilinear")
    img = tf.clip_by_value(img, 0.0, 255.0)
    return img

