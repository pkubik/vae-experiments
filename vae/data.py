from collections import namedtuple
from contextlib import suppress
from pathlib import Path

import tensorflow as tf
import yaml

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


def default_config() -> dict:
    path = Path(vae.__file__).parent.parent / 'configs'
    path.mkdir(parents=True, exist_ok=True)

    # noinspection PyUnusedLocal
    config = None
    with suppress(FileNotFoundError):
        config = yaml.load((path / 'local.yml').open())
    if config is None:
        with suppress(FileNotFoundError):
            config = yaml.load((path / 'default.yml').open())
    if config is None:
        config = {}

    return config


def save_yaml(content: dict, path: Path):
    yaml.dump(content, path.open('w'))
