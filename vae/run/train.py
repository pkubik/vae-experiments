import sys
import vae.configurator as cfg
from vae.data import load_mnist, default_models_dir, default_config, save_yaml
from vae.model import Model


def train(model_name: str):
    with cfg.use(default_config()):
        data = load_mnist()

        model = Model(default_models_dir() / model_name)
        model.train(data.train)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Model name not specified!")
        exit(-1)

    train(sys.argv[1])
