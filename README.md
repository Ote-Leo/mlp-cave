# MLP Cave

A minimalistic Multi-Layer Perceptron (MLP) implementation. Check the
[Implementation instructions](./instructions.md) for more details.

## Usage Examples

### XOr Gate

```python
>>> import numpy as np
>>> import mlp
>>>
>>> DATA = (
>>>     (np.array([0, 0]), np.array([0])),
>>>     (np.array([0, 1]), np.array([1])),
>>>     (np.array([1, 0]), np.array([1])),
>>>     (np.array([1, 1]), np.array([0])),
>>> )
>>>
>>> network = mlp.Network((2, 2, 1))
>>> mlp.train(network, DATA)
>>>
>>> for input, expected in DATA:
>>>     x = input[0]
>>>     y = input[1]
>>>     print(f"{x} ^ {y} = {network.forward(input)[0]:.03f}; {expected[0]:.03f}")
0 ^ 0 = 0.019; 0.000
0 ^ 1 = 0.977; 1.000
1 ^ 0 = 0.977; 1.000
1 ^ 1 = 0.024; 0.000
```
