from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

try:
    from collections.abc import Sequence
except ImportError:
    from typing import Sequence

Weights: TypeAlias = NDArray
Biases: TypeAlias = NDArray
LearningRate: TypeAlias = np.float64

InputData: TypeAlias = NDArray
ExpectedOutput: TypeAlias = NDArray
Batch: TypeAlias = Sequence[tuple[InputData, ExpectedOutput]]
