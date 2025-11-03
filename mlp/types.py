from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

Weights: TypeAlias = NDArray
Biases: TypeAlias = NDArray
LearningRate: TypeAlias = np.float64

Patterns: TypeAlias = NDArray
"""A 2-D NumPy array of shape (N, input_dim)."""
Labels: TypeAlias = NDArray
