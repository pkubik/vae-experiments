from collections import namedtuple
from pathlib import Path

import tensorflow as tf
import vae


Dataset = namedtuple('Dataset', 'images, labels')
Split = namedtuple('Split', 'train, test')


def load_mnist() -> Split:
    train_tuple, test_tuple = tf.keras.datasets.mnist.load_data()
    return Split(Dataset(*train_tuple), Dataset(*test_tuple))


def default_models_dir() -> Path:
    path = Path(vae.__file__).parent.parent / 'models'
    path.mkdir(parents=True, exist_ok=True)
    return path
