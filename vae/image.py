import numpy as np


def norm_images(images: np.ndarray) -> np.ndarray:
    return images.astype(np.float32) / 255
