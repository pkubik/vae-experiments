from collections import namedtuple
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


def open_config(name: str = 'local') -> dict:
    path = Path(vae.__file__).parent.parent / 'configs' / '{}.yml'.format(name)
    return yaml.load(path.open())


def save_config(content: dict, path: Path):
    configs_path = Path(vae.__file__).parent.parent / 'configs'
    configs_path.mkdir(parents=True, exist_ok=True)
    yaml.dump(content, (configs_path / 'latest.yml').open('w'))
    yaml.dump(content, path.open('w'))
