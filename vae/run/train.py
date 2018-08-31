import sys

from contextlib import ExitStack
from pathlib import Path

import vae.configurator as cfg
from vae.data import load_mnist, default_models_dir, open_config
from vae.model import Model


def create_latest_model_symlink(model_path: Path):
    models_dir = default_models_dir()
    latest_path = model_path / 'symlink'
    latest_path.symlink_to(model_path, target_is_directory=True)
    latest_path.replace(models_dir / 'latest')


def train(model_name: str, config_name: str = None):
    with ExitStack() as stack:
        if config_name:
            stack.enter_context(cfg.use(open_config(config_name)))

        data = load_mnist()

        model_path = default_models_dir() / model_name
        model = Model(model_path)
        create_latest_model_symlink(model_path)
        model.train(data.train)


def main():
    if len(sys.argv) < 2:
        print("Model name not specified!")
        exit(-1)
    model_name = sys.argv[1]

    config_name = None
    if len(sys.argv) == 3:
        config_name = sys.argv[-1]

    train(model_name, config_name)


if __name__ == '__main__':
    main()
