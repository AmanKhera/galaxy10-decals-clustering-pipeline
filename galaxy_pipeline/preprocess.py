# galaxy_pipeline/preprocess.py
import tensorflow as tf

def load_image_rgb(path: str, size=(256, 256)) -> tf.Tensor:
    """
    Loads an RGB image from disk and returns a float32 tensor in 0..255 range.
    Output shape: (H, W, 3)
    """
    img_bytes = tf.io.read_file(path)

    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)

    img = tf.image.resize(img, size, method="bilinear")
    img = tf.cast(img, tf.float32)

    img = tf.clip_by_value(img, 0.0, 255.0)

    return img


def make_batch(img: tf.Tensor) -> tf.Tensor:
    """
    Adds batch dimension.
    Input:  (H, W, 3)
    Output: (1, H, W, 3)
    """
    return img[tf.newaxis, ...]
