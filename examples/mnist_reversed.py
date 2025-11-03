import logging
import math

import numpy as np

import mlp
from mlp.loaders import load_mnist_reversed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

shape = (3, 16, 16, 1)

layer_descriptors = (
    mlp.LayerDescriptor(size=3, activation_function=mlp.Sigmoid()),
    mlp.LayerDescriptor(size=16, activation_function=mlp.Sigmoid()),
    mlp.LayerDescriptor(size=16, activation_function=mlp.Sigmoid()),
    mlp.LayerDescriptor(size=1, activation_function=mlp.Sigmoid()),
)

network = mlp.Network(layer_descriptors)

data, labels = load_mnist_reversed("./datasets/MNIST_ORG")

error_threshold = 1e-5
report_every = 100


learning_rates = (
    (1e-1, 10),
    (1e-3, 1e-2),
)

batch_sizes = ((1e-1, 64),)

for learning_rate, batch_size in batch_sizes:
    logging.info(f"passed first phase, decrementing learning rates to {learning_rate}")
    for layer in network.layers:
        layer.learning_rate = np.float64(learning_rate)

    logging.info(f"training on batches of size {batch_size}")
    mlp.train_batched(
        network,
        data,
        labels,
        batch_size=batch_size,
        error_threshold=0.01,
    )
    backup_network_path = f"mnist-reversed-{batch_size}batched.mlp"
    logging.info(f"saving a backup version to {backup_network_path}")
    network.dump(backup_network_path)

err = math.inf
count = 0
pattern_legnth, labels_length = len(data), len(labels)
new_learning_rate = learning_rates[1][0]
logging.info(f"passed first phase, decrementing learning rates to {new_learning_rate}")
for layer in network.layers:
    layer.learning_rate = np.float64(new_learning_rate)

while err > learning_rates[1][1]:
    if count % report_every == 0:
        logging.info(f"Step {count:6d} | Loss = {err:.06f}")
    idx = count % pattern_legnth

    pattern, label = data[idx], labels[idx]
    network.train_pattern(pattern, label)
    err = mlp.loss(network, (data, labels))
    if count > 0 and count % 10_000 == 0:
        logging.info("saving a backup to 'mnist-reversed.backup.mlp'")
        network.dump("mnist-reversed.backup.mlp")
    count += 1


network.dump("mnist-reversed.mlp")
