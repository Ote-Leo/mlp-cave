import gzip
import re
import struct
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def _read_idx_images(path: str | Path, flatten: bool = True) -> NDArray[np.float64]:
    """Read MNIST image file (idx3) into a normalized float64 NumPy array."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid magic number {magic} in {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
        images = (
            data.reshape(num, rows * cols) if flatten else data.reshape(num, rows, cols)
        )
        images = images.astype(np.float64) / 255.0
    return images


def _read_idx_labels(path: str | Path) -> NDArray[np.uint8]:
    """Read MNIST label file (idx1) into a uint8 NumPy array."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        magic, _ = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid magic number {magic} in {path}")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


def _one_hot(labels: NDArray[np.uint8], num_classes: int = 10) -> NDArray[np.float64]:
    """Convert label vector to one-hot encoded matrix."""
    out = np.zeros((labels.size, num_classes), dtype=np.float64)
    out[np.arange(labels.size), labels] = 1.0
    return out


def load_mnist(
    root: str | Path,
    train: bool = True,
    one_hot: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Load MNIST dataset from `root` directory.

    Args:
        root:
            Directory containing MNIST files
            (e.g. train-images-idx3-ubyte[.gz], train-labels-idx1-ubyte[.gz]).
        train:
            Whether to load training or test split.
        one_hot:
            Whether to one-hot encode the labels.

    Returns
        images : (N, 784) float64 ndarray
            Normalized image data in [0, 1].
        labels : (N,) or (N, 10) float64 ndarray
            Labels, one-hot encoded if requested.
    """
    root = Path(root)
    prefix = "train" if train else "t10k"

    img_path = next(root.glob(f"{prefix}-images.idx3-ubyte*"))
    lbl_path = next(root.glob(f"{prefix}-labels.idx1-ubyte*"))

    images = _read_idx_images(img_path)
    labels = _read_idx_labels(lbl_path)
    if one_hot:
        labels = _one_hot(labels)
    labels = labels.astype(np.float64)
    return images, labels


def load_mnist_reversed(
    root: str | Path,
    train: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    root = Path(root)
    prefix = "train" if train else "t10k"

    img_path = next(root.glob(f"{prefix}-images.idx3-ubyte*"))
    lbl_path = next(root.glob(f"{prefix}-labels.idx1-ubyte*"))

    all_images = _read_idx_images(img_path, flatten=False)
    labels = _read_idx_labels(lbl_path).astype(np.float64)

    images = []

    for i in range(10):
        for j in range(len(labels)):
            if i == labels[j]:
                images.append(all_images[j])
                break
    assert len(images) == 10, len(images)

    training_data = []
    training_data_labels = []

    for i, image in enumerate(images):
        for j, row in enumerate(image):
            for k, col in enumerate(row):
                x = k / 28
                y = j / 28
                training_data.append([x, y, i])
                training_data_labels.append([col])

    return np.array(training_data), np.array(training_data_labels)


COMMENT_LINE_PATTERN = re.compile(r"^\s*#.*$")
SHAPE_LINE_PATTERN = re.compile(
    r"^\s*#\s*P\s*=\s*(\d+)\s*N\s*=\s*(\d+)\s*M\s*=\s*(\d+).*$"
)
DATA_LINE_PATTERN = re.compile(r"^\s*([^#].*?)\s*(?:#.*)?$")
NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


def parse_numbers_from_line(line: str) -> list[float]:
    """Extract a list of floating point numbers from a data line."""
    if not (match := DATA_LINE_PATTERN.match(line)):
        return []
    data_str = match.group(1)
    return list(map(float, NUMBER_PATTERN.findall(data_str)))


def load_basic_raw(path: str | Path) -> NDArray[np.float64]:
    """Load raw numeric data from the file, igonoring comments and extracting
    any numbers found.

    Returns:
        A NumPy aray of shape (P, N+M) where P is the number patterns.
    """
    data_lines = []

    with open(path, "r") as file:
        for line in filter(None, map(str.strip, file)):
            if COMMENT_LINE_PATTERN.match(line):
                continue
            if not (numbers := parse_numbers_from_line(line)):
                continue  # mostly likely unreachable
            data_lines.append(numbers)

    return np.array(data_lines, dtype=np.float64)


def load_basic(path: str | Path) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Load structured data with shape comment and split into patterns and
    labels.

    Returns:
        - patterns: shape (P, N)
        - labels: shape (P, M)
    """
    patterns, labels = [], []
    patterns_shape, labels_shape = None, None

    with open(path, "r") as file:
        for i, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            if COMMENT_LINE_PATTERN.match(line):
                match = SHAPE_LINE_PATTERN.match(line)
                if match and not (patterns_shape and labels_shape):
                    _, patterns_shape, labels_shape = map(int, match.groups())
                continue

            if patterns_shape is None or labels_shape is None:
                raise ValueError("Missing Shape Comment Line")

            numbers = parse_numbers_from_line(line)
            if len(numbers) != patterns_shape + labels_shape:
                raise ValueError(
                    f"Expected {patterns_shape + labels_shape} values per line, "
                    f"got {len(numbers)} at line {i}: {line}"
                )

            patterns.append(numbers[:patterns_shape])
            labels.append(numbers[patterns_shape:])

    patterns = np.array(patterns, dtype=np.float64)
    labels = np.array(labels, dtype=np.float64)
    return patterns, labels
