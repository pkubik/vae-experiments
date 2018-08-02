import numpy as np


def norm_images(images: np.ndarray) -> np.ndarray:
    return np.expand_dims(images.astype(np.float32) / 255, -1)
