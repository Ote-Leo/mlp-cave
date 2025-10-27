"""Minimal feedforward neural network with backpropagation using NumPy.

This module implements a simple fully connected neural network (multi-layer
perceptron) with support for different activation functions, including
Sigmoid, Tanh, ReLU, and Identity. It supports training with
pattern-by-pattern backpropagation and calculates the mean squared error loss.
"""

import io
import itertools
import logging
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

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


HEADER_SIGNATURE: bytes = b"MLPN"  # MLP Network
VERSION: bytes = (1).to_bytes(1, "little")
COMPRESSED: bytes = (1).to_bytes(1, "little")
UNCOMPRESSED: bytes = (0).to_bytes(1, "little")


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

    def _dump(self) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        data_list = []
        shapes = []
        act_names = []
        learning_rates = []

        for layer in self.layers:
            data_list.append(layer.weights.ravel())
            data_list.append(layer.biases.ravel())
            shapes.append(layer.weights.shape)
            act_names.append(layer.activation_function.__class__.__name__)
            learning_rates.append(layer.learning_rate)

        all_data = np.concatenate(data_list)
        shapes_arr = np.array(shapes, dtype=np.int32).flatten()
        act_names_arr = np.array(act_names, dtype="S")
        learning_rates_arr = np.array(learning_rates, dtype=np.float64)

        return all_data, shapes_arr, act_names_arr, learning_rates_arr

    def _dumps(self) -> bytes:
        f = io.BytesIO()
        all_data, shapes_arr, act_names_arr, learning_rates_arr = self._dump()
        np.savez(
            f,
            data=all_data,
            shapes=shapes_arr,
            act_names=act_names_arr,
            learning_rates=learning_rates_arr,
            allow_pickle=False,
        )
        return f.getvalue()

    def dumps(self, compress: bool = True) -> bytes:
        """Serialize network to bytes."""
        data = self._dumps()

        header = HEADER_SIGNATURE + VERSION
        if compress:
            LOGGER.info("zlib compressing Network serialization")
            data = zlib.compress(data)
            header += COMPRESSED
        else:
            header += UNCOMPRESSED
        data = header + data

        crc = zlib.crc32(data) & 0xFFFF_FFFF
        checksum = struct.pack(">I", crc)

        return data + checksum

    def dump(self, path: str | Path, compress: bool = True):
        """Save network to file."""
        data = self.dumps(compress=compress)
        path = Path(path)
        path.write_bytes(data)

    @classmethod
    def _load(
        cls,
        all_data: NDArray,
        shapes_arr: NDArray,
        act_names_arr: NDArray,
        learning_rates_arr: NDArray,
    ):
        shapes = shapes_arr.reshape(-1, 2)
        act_names = [name.tobytes().decode("utf-8") for name in act_names_arr]
        layers = []
        offset = 0

        for shape, act_name, lr in zip(shapes, act_names, learning_rates_arr):
            w_size = shape[0] * shape[1]
            b_size = shape[0]
            w = all_data[offset : offset + w_size].reshape(shape)
            offset += w_size
            b = all_data[offset : offset + b_size]
            offset += b_size
            act = {
                "Sigmoid": Sigmoid,
                "Tanh": Tanh,
                "ReLU": ReLU,
                "Identity": Identity,
            }[act_name]()
            layers.append(Layer(w, b, lr, act))

        net = cls([shape[1] for shape in shapes] + [shapes[-1][0]])
        net.layers = layers
        return net

    @classmethod
    def _loads(cls, raw: bytes) -> "Network":
        with io.BytesIO(raw) as f:
            npz = np.load(f)
            data = npz["data"]
            shapes = npz["shapes"]
            act_names = npz["act_names"]
            learning_rates = npz["learning_rates"]
        return cls._load(data, shapes, act_names, learning_rates)

    @classmethod
    def loads(cls, data_bytes: bytes) -> "Network":
        """Deserialize network from bytes."""
        LOGGER.debug("Checking header signature")
        if not data_bytes.startswith(HEADER_SIGNATURE):
            raise ValueError("Invalid file header")

        LOGGER.debug("Checking CRC32 Checksum")
        payload, checksum_bytes = data_bytes[:-4], data_bytes[-4:]
        given_checksum = struct.unpack(">I", checksum_bytes)[0]
        computed_checksum = zlib.crc32(payload) & 0xFFFF_FFFF
        if given_checksum != computed_checksum:
            raise ValueError(
                f"Checksum mismatch: expected {given_checksum:#x}, "
                f"computed {computed_checksum:#x}. Corrupted or incomplete file"
            )

        LOGGER.debug("Checking header version")
        version = data_bytes[len(HEADER_SIGNATURE)]
        if version != VERSION[0]:
            raise ValueError(f"Unsupported version {version}")

        LOGGER.debug("Checking compression flag")
        compression_flag = data_bytes[len(HEADER_SIGNATURE) + 1]
        data = data_bytes[len(HEADER_SIGNATURE) + 2 : -4]
        if compression_flag == COMPRESSED[0]:
            LOGGER.info("zlib decompressing data")
            data = zlib.decompress(data)
        elif compression_flag != UNCOMPRESSED[0]:
            raise ValueError(f"Unknown compression flag {compression_flag!r}")

        return Network._loads(data)

    @classmethod
    def load(cls, path: str | Path) -> "Network":
        """Load network from file."""
        raw_data = Path(path).read_bytes()
        return cls.loads(raw_data)


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
