"""Minimal feedforward neural network with backpropagation using NumPy.

This module implements a simple fully connected neural network (multi-layer
perceptron) with support for different activation functions, including
Sigmoid, Tanh, ReLU, and Identity. It supports training with
pattern-by-pattern backpropagation and calculates the mean squared error loss.
"""

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
    """Generate a NumPy array of given shape with uniform random values between min and max.

    Args:
        shape: Desired shape of the array.
        max: Maximum value (inclusive).
        min: Minimum value (inclusive).

    Returns:
        Array filled with uniform random values in [min, max].
    """
    return np.random.uniform(min, max, shape).astype(np.float64)


DEFAULT_LEARNING_RATE: float = 1e-1


class ActivationFunction:
    """Base class for activation functions."""

    def __call__(self, arr: NDArray) -> NDArray:
        """Computes the activation function output for the input array."""
        raise NotImplementedError()

    def derivative(self, arr: NDArray) -> NDArray:
        """Computes the derivative of the activation function for backpropagation."""
        raise NotImplementedError()


class Sigmoid(ActivationFunction):
    r"""Sigmoid activation function: :math:`f(x) = \frac{1}{1 + e^{-x}}`.

    The derivative is :math:`f'(x) = f(x) \dot (1 - f(x))`.
    Includes overflow protection for large negative values.
    """

    def __call__(self, arr: NDArray) -> NDArray:
        return np.where(
            arr >= 0,
            1 / (1 + np.exp(-arr)),
            np.exp(arr) / (1 + np.exp(arr)),  # avoiding overflow large negative numbers
        )

    def derivative(self, arr: NDArray) -> NDArray:
        return self(arr) * (1 - self(arr))


class Tanh(ActivationFunction):
    r"""Hyperbolic tangent activation function: :math:`f(x) = \tanh{x}`.

    The derivative is :math:`f'(x) = 1 - f(x)^2`.
    """

    def __call__(self, arr: NDArray) -> NDArray:
        return np.tanh(arr)

    def derivative(self, arr: NDArray) -> NDArray:
        return 1 - self(arr) ** 2


class Identity(ActivationFunction):
    """Identity (linear) activation function: :math:`f(x) = x`.

    The derivative is :math:`f'(x) = 1`.
    """

    def __call__(self, arr: NDArray) -> NDArray:
        return arr

    def derivative(self, arr: NDArray) -> NDArray:
        return np.ones_like(arr)


class ReLU(ActivationFunction):
    r"""Rectified Linear Unit (ReLU) activation function: :math:`f(x) = \max(0, x)`.

    The derivative is :math:`f'(n) = \begin{cases}
    1 & \text{if } x > 0 \\
    0 & \text{otherwise}
    \end{cases}`
    """

    def __call__(self, arr: NDArray) -> NDArray:
        return np.maximum(0, arr)

    def derivative(self, arr: NDArray) -> NDArray:
        return (arr > 0).astype(np.float64)


@dataclass
class LayerDescriptor:
    """Descriptor for a neural network layer (Layer)."""

    size: int
    """Number of neurons in the layer."""
    learning_rate: float = DEFAULT_LEARNING_RATE
    """Learning rate used for updating weights and biases."""
    activation_function: ActivationFunction = Sigmoid()
    """Activation function used in the layer."""


Weights = NDArray
Biases = NDArray
LearningRate = np.float64


@dataclass
class Layer:
    """Represents a fully connected layer in a neural network."""

    weights: Weights
    """Weight matrix of shape (`next_layer_size`, `previous_layer_size`)."""
    biases: Biases
    """Bias vector for the layer."""
    learning_rate: LearningRate
    """Learning rate for weight and bias updates."""
    activation_function: ActivationFunction
    """Activation function used for this layer."""


InputData = NDArray
ExpectedOutput = NDArray
Batch = Sequence[tuple[InputData, ExpectedOutput]]


class Network:
    """Multi-Layer Perceptron (MLP) neural network."""

    def __init__(
        self,
        description: Sequence[int | LayerDescriptor],
        seed: float | None = None,
    ):
        """Initialize a multi-layer perceptron network.

        Args:
            description: Layer sizes and descriptors to define the network architecture.
            seed: An optional random seed for reproducible initialization.

        Notes:
            - Each layer is initialized with random weights and biases in the range [-2, 2].
        """
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
        """A tuple of layer sizes, including input and output layers."""
        s = [self.layers[0].weights.shape[1]]
        for layer in self.layers:
            cur_shape, _ = layer.weights.shape
            s.append(cur_shape)
        return tuple(s)

    def forward(self, input_data: InputData) -> NDArray:
        """Computes forward pass through the network for given input."""
        acc = input_data
        for i, layer in enumerate(self.layers):
            LOGGER.debug(
                f"Layer {i} forward | "
                f"input shape={acc.shape} | "
                f"weights shape={layer.weights.shape} | "
                f"biases shape={layer.biases.shape}"
            )

            mult = layer.weights @ acc.T + layer.biases
            acc = layer.activation_function(mult)
        return acc

    def forward_train(
        self,
        input_data: InputData,
    ) -> tuple[list[NDArray], list[NDArray]]:
        r"""Compute forward pass and return activations and derivatives for backpropagation.

        Returns:
            activations: List of activations for each layer, including input.
            derivs: List of derivatives of each layer’s weighted input.
        """
        activations = [input_data]
        derivs = []

        acc = input_data
        for i, layer in enumerate(self.layers):
            LOGGER.debug(
                f"Layer {i} forward | "
                f"input shape={acc.shape} | "
                f"weights shape={layer.weights.shape} | "
                f"biases shape={layer.biases.shape}"
            )

            mult = layer.weights @ acc.T + layer.biases
            acc = layer.activation_function(mult)
            activations.append(acc)
            derivs.append(layer.activation_function.derivative(mult))

        return activations, derivs

    def train_pattern(
        self,
        input_data: InputData,
        expected: ExpectedOutput,
    ):
        """Performs a single pattern update using backpropagation."""
        activations, derivs = self.forward_train(input_data)
        deltas = []

        # Output layer delta
        error = expected - activations[-1]
        deltas.append(error * derivs[-1])

        # Hidden layer deltas
        for layer, deriv in zip(reversed(self.layers), reversed(derivs[:-1])):
            deltas.append((layer.weights.T @ deltas[-1]) * deriv.ravel())

        # Update weights and biases
        for delta, layer, activation in zip(reversed(deltas), self.layers, activations):
            layer.weights += layer.learning_rate * (
                delta[:, None] * activation[None, :]
            )
            layer.biases += layer.learning_rate * delta


def loss(
    network: Network,
    batch: Batch,
) -> np.float64:
    """Computes the total mean squared error over a batch of input-output pairs.

    Args:
        network: The neural network to evaluate.
        batch: A sequence of (input, expected output) pairs.

    Returns:
        Total mean squared error of the network on the batch.
    """
    total_err: np.float64 = np.float64(0)
    for input_data, expected_output in batch:
        res = network.forward(input_data)
        diff = expected_output - res
        total_err += (diff * diff).sum()
    return np.float64(0.5) * total_err


DEFAULT_ERROR_THRESHOLD: float = 1e-3


def train(
    network: Network,
    batch: Batch,
    iteration_count: int | None = None,
    error_threshold: float = DEFAULT_ERROR_THRESHOLD,
    verbose: bool = True,
    report_every: int = 100,
):
    """Train a neural network over a batch of patterns.

    Args:
        network: Neural network instance to train.
        batch: A sequence of (input, expected output) training pairs.
        iteration_count: Maximum number of training iterations. If None, training continues until error_threshold is reached.
        error_threshold: Early stopping threshold for loss.
        verbose: If True, log progress at regular intervals.
        report_every: Interval of steps at which progress is logged.
    """
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
