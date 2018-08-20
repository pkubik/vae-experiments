import sys

from contextlib import ExitStack

import vae.configurator as cfg
from vae.data import load_mnist, default_models_dir, open_config
from vae.model import Model


def train(model_name: str, config_name: str = None):
    with ExitStack() as stack:
        if config_name:
            stack.enter_context(cfg.use(open_config(config_name)))

        data = load_mnist()

        model = Model(default_models_dir() / model_name)
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
