import logging

import mlp
from mlp.loaders import load_mnist

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

IMAGE_WIDTH: int = 28
IMAGE_HEIGHT: int = IMAGE_WIDTH
IMAGE_SIZE: int = IMAGE_WIDTH * IMAGE_HEIGHT

shape = (IMAGE_SIZE, 16, 16, 10)

network = mlp.Network(shape)

data, labels = load_mnist("./datasets/mnist")
mlp.train_batched(network, data, labels)
network.dump("mnist.mlp")
