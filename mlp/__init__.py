import itertools
import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

try:
    from collections.abc import Sequence
except ImportError:
    from typing import Sequence

LOGGER = logging.getLogger(__name__)

def random_array(
    shape: int | tuple[int, ...],
    max: float = 2.0,
    min: float = -2.0,
) -> NDArray[np.float64]:
    return np.random.uniform(min, max, shape).astype(np.float64)


DEFAULT_LEARNING_RATE: float = 1e-1


class ActivationFunction:
    def __call__(self, arr: NDArray) -> NDArray:
        raise NotImplementedError()

    def derivative(self, arr: NDArray) -> NDArray:
        raise NotImplementedError()


class Sigmoid(ActivationFunction):
    def __call__(self, arr: NDArray) -> NDArray:
        return np.where(
            arr >= 0,
            1 / (1 + np.exp(-arr)),
            np.exp(arr) / (1 + np.exp(arr)),  # avoiding overflow large negative numbers
        )

    def derivative(self, arr: NDArray) -> NDArray:
        return self(arr) * (1 - self(arr))


class Tanh(ActivationFunction):
    def __call__(self, arr: NDArray) -> NDArray:
        return np.tanh(arr)

    def derivative(self, arr: NDArray) -> NDArray:
        return 1 - self(arr) ** 2


class Identity(ActivationFunction):
    def __call__(self, arr: NDArray) -> NDArray:
        return arr

    def derivative(self, arr: NDArray) -> NDArray:
        return np.ones_like(arr)


class ReLU(ActivationFunction):
    def __call__(self, arr: NDArray) -> NDArray:
        return np.maximum(0, arr)

    def derivative(self, arr: NDArray) -> NDArray:
        return (arr > 0).astype(np.float64)


@dataclass
class LayerDescriptor:
    size: int
    learning_rate: float = DEFAULT_LEARNING_RATE
    activation_function: ActivationFunction = Sigmoid()


Weights = NDArray
Biases = NDArray
LearningRate = np.float64


@dataclass
class Layer:
    weights: Weights
    biases: Biases
    learning_rate: LearningRate
    activation_function: ActivationFunction


class Network:
    def __init__(
        self,
        description: Sequence[int | LayerDescriptor],
        seed: float | None = None,
    ):
        if seed is not None:
            LOGGER.debug(f"Setting random seed to {seed}")
            np.random.seed(seed)

        self.layers: list[Layer] = []

        descs = tuple(
            desc if isinstance(desc, LayerDescriptor) else LayerDescriptor(desc)
            for desc in description
        )

        for prev_desc, next_desc in zip(descs, descs[1:]):
            weights = random_array((next_desc.size, prev_desc.size))
            biases = random_array(next_desc.size)
            layer = Layer(
                weights,
                biases,
                np.float64(next_desc.learning_rate),
                next_desc.activation_function,
            )
            self.layers.append(layer)


        for i, layer in enumerate(self.layers):
            weights = layer.weights
            biases = layer.biases
            LOGGER.info(
                f"Layer {i} initialized | "
                f"weights shape={weights.shape} biases shape={biases.shape}"
            )
            LOGGER.debug(f"Weight min={weights.min():.2f}, max={biases.max():.2f}")

    @property
    def shape(self) -> tuple[int, ...]:
        s = [self.layers[0].weights.shape[1]]
        for layer in self.layers:
            cur_shape, _ = layer.weights.shape
            s.append(cur_shape)
        return tuple(s)

    def forward(self, input_data: NDArray) -> NDArray:
        acc = input_data
        for i, layer in enumerate(self.layers):
            LOGGER.debug(
                f"Layer {i} forward | "
                f"input shape={acc.shape} | "
                f"weights shape={layer.weights.shape} | "
                f"biases shape={layer.biases.shape}"
            )
            expected_out_shape = (layer.weights.shape[0], acc.shape[1])
            LOGGER.debug(f"Expected output shape={expected_out_shape}")

            mult = layer.weights @ acc.T + layer.biases
            acc = layer.activation_function(mult)
        return acc

    def forward_train(
        self,
        input_data: NDArray,
    ) -> tuple[list[NDArray], list[NDArray]]:
        r"""
        Returns
        -------

        - The intermediate output of each layer including the input layer :math:`\widetilde{out_k}`.
        - The intermediate transfer function over the weighted summation :math:`f'(\text{net}_k)`
        """
        outs = [input_data]
        derive_activation = []

        acc = input_data
        for i, layer in enumerate(self.layers):
            LOGGER.debug(
                f"Layer {i} forward | "
                f"input shape={acc.shape} | "
                f"weights shape={layer.weights.shape} | "
                f"biases shape={layer.biases.shape}"
            )
            expected_out_shape = (layer.weights.shape[0], acc.shape[1])
            LOGGER.debug(f"Expected output shape={expected_out_shape}")

            mult = layer.weights @ acc.T + layer.biases
            acc = layer.activation_function(mult)
            outs.append(acc)
            derive_activation.append(layer.activation_function.derivative(mult))

        return outs, derive_activation

    def train_pattern(
        self,
        input_data: NDArray,
        expected: NDArray,
    ):
        new_layers = [
            [np.copy(layer.weights), np.copy(layer.biases)] for layer in self.layers
        ]
        activations, derivs = self.forward_train(input_data)

        # Output layer
        output_layer = self.layers[-1]
        diff_output = expected - activations[-1]
        delta = diff_output * derivs[-1]

        new_layers[-1][0] += output_layer.learning_rate * (
            delta[:, None] @ activations[-2][None, :]
        )
        new_layers[-1][1] += output_layer.learning_rate * delta

        # Hidden layers
        for l in range(len(self.layers) - 2, -1, -1):
            layer = self.layers[l]
            next_layer = self.layers[l + 1]
            delta = (next_layer.weights.T @ delta) * derivs[l].ravel()
            new_layers[l][0] += layer.learning_rate * (
                delta[:, None] * activations[l][None, :]
            )
            new_layers[l][1] += layer.learning_rate * delta

        for layer, (weights, biases) in zip(self.layers, new_layers):
            layer.weights = weights
            layer.biases = biases


InputData = NDArray
ExpectedOutput = NDArray
Batch = Sequence[tuple[InputData, ExpectedOutput]]


def loss(
    network: Network,
    batch: Batch,
) -> np.float64:
    total_err: np.float64 = np.float64(0)
    for input_data, expected_output in batch:
        res = network.forward(input_data)
        diff = expected_output - res
        total_err += (diff * diff).sum()
    return np.float64(0.5) * total_err


DEAFULT_ERROR_THRESHOLD: float = 1e-3


def train(
    network: Network,
    batch: Batch,
    iteration_count: int | None = None,
    error_threshold: float = DEAFULT_ERROR_THRESHOLD,

    verbose: bool = True,
    report_every: int = 100,
):
    import math

    batches = itertools.cycle(batch)

    err = math.inf
    count = 0
    while True:
        if iteration_count is not None:
            if count >= iteration_count:
                break
        elif err < error_threshold:
            break

        if verbose and count % report_every == 0:
            LOGGER.info(f"Step {count:6d} | Loss = {err:.06f}")
        count += 1

        input_data, expected_output = next(batches)
        network.train_pattern(input_data, expected_output)
        err = loss(network, batch)
