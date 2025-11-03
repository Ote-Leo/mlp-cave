import logging

import mlp
from mlp.loaders import load_mnist_reversed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

shape = (3, 16, 16, 1)

network = mlp.Network(shape)

data, labels = load_mnist_reversed("datasets/mnist")
mlp.train_batched(network, data, labels)
network.dump("mnist-reversed.mlp")
