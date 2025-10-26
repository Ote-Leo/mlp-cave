import itertools
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

try:
    from collections.abc import Sequence
except ImportError:
    from typing import Sequence


def random_array(
    shape: int | tuple[int, ...],
    max: float = 2.0,
    min: float = -2.0,
) -> NDArray[np.float64]:
    return np.random.uniform(min, max, shape).astype(np.float64)


DEFAULT_LEARNING_RATE: float = 1e-3


class ActivationFunction:
    def __call__(self, arr: NDArray) -> NDArray:
        raise NotImplementedError()

    def derivative(self, arr: NDArray) -> NDArray:
        raise NotImplementedError()


class Sigmoid(ActivationFunction):
    def __call__(self, arr: NDArray) -> NDArray:
        return 1 / (1 + np.exp(-arr))

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
Layer = tuple[Weights, Biases, LearningRate, ActivationFunction]


class Network:
    def __init__(
        self,
        description: Sequence[int | LayerDescriptor],
    ):
        self.layers: list[Layer] = []

        descs = tuple(
            desc if isinstance(desc, LayerDescriptor) else LayerDescriptor(desc)
            for desc in description
        )

        for prev_desc, next_desc in zip(descs, descs[1:]):
            weights = random_array((next_desc.size, prev_desc.size))
            biases = random_array(next_desc.size)
            layer = (
                weights,
                biases,
                np.float64(next_desc.learning_rate),
                next_desc.activation_function,
            )
            self.layers.append(layer)

    @property
    def shape(self) -> tuple[int, ...]:
        s = [self.layers[0][0].shape[1]]
        for layer in self.layers:
            cur_shape, _ = layer[0].shape
            s.append(cur_shape)
        return tuple(s)

    def forward(self, input_data: NDArray) -> NDArray:
        acc = input_data
        for weights, biases, _, activation_function in self.layers:
            mult = weights @ acc.T + biases
            acc = activation_function(mult)
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
        for weights, biases, _, activation_function in self.layers:
            mult = weights @ acc.T + biases
            acc = activation_function(mult)
            outs.append(acc)
            derive_activation.append(activation_function.derivative(mult))

        return outs, derive_activation

    def train_pattern(
        self,
        input_data: NDArray,
        expected: NDArray,
    ):
        new_layers = [
            [np.copy(weights), np.copy(biases)] for weights, biases, *_ in self.layers
        ]
        outs, derive_activation = self.forward_train(input_data)

        # Output layer
        output_weights, _, learning_rate, *_ = self.layers[-1]
        diff_output = expected - outs[-1]
        delta_output = diff_output * derive_activation[-1]
        delta = (delta_output.reshape(-1, 1) * output_weights).sum(axis=0)
        new_layers[-1][0] += learning_rate * (delta_output[:, None] * outs[-2][None, :])
        new_layers[-1][1] += learning_rate * delta_output

        # Hidden layers
        hidden_layers = zip(reversed(self.layers[:-1]), reversed(outs[:-2]))
        for i, (layer, layer_input) in enumerate(hidden_layers, start=2):
            weights, _, learning_rate, *_ = layer
            current_delta = np.copy(delta)
            delta = (current_delta.reshape(-1, 1) * weights).sum(axis=0)
            new_layers[-i][0] += learning_rate * (
                current_delta[:, None] * layer_input[None, :]
            )
            new_layers[-i][1] += learning_rate * current_delta

        # Update weights and biases
        for i, (layer, new_weights) in enumerate(zip(self.layers, new_layers)):
            _, _, learning_rate, activation_function = layer
            weights, biases = new_weights
            self.layers[i] = (weights, biases, learning_rate, activation_function)


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
        total_err = (diff * diff).sum()
    return np.float64(0.5) * total_err


DEAFULT_ERROR_THRESHOLD: float = 1e-3


def train(
    network: Network,
    batch: Batch,
    iteration_count: int | None = None,
    error_threshold: float = DEAFULT_ERROR_THRESHOLD,
):
    import math

    batches = itertools.cycle(batch)

    err = math.inf
    count = 0
    while True:
        if iteration_count is not None:
            if count >= iteration_count:
                break
            count += 1
        elif err < error_threshold:
            break

        input_data, expected_output = next(batches)
        network.train_pattern(input_data, expected_output)
        err = loss(network, batch)
