import sys
from vae.data import load_mnist, default_models_dir
from vae.model import Model


def train(model_name: str):
    data = load_mnist()

    model = Model(default_models_dir() / model_name)
    #model.num_epochs = 10
    model.train(data.train)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Model name not specified!")
        exit(-1)

    train(sys.argv[1])
