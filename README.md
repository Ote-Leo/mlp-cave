# Multi-Layer Perceptrons

Implement in Python a Multi-Layer Perzeptron including the learning rule
Backpropagation of Error and test your program. Do not use libraries that have
pre-implemented neural network structures.

## Implement a Multi Layer Perceptron

Your Program must be capable of setting the structure of the network, number of
layers (maximum 4), number of neurons per layer (maximum 1000),
transferfunction separately for each layer (_tanh_ or _logistic_ or
_identity_). It is O.K. to set these parameters directly within the source
code, please do not implement a user interface for that.

Initialize the weights to random values between -2.0 and +2.0, make sure that
the random number generator is under your control, and that you can reproduce
your results. Set/initialize the random number generator explicitly (random
seed).

## Implement Backpropagation of Error

Implement the 7 steps of Backpropagation of Error Algorithm (from the lecture).
Your program shall read training patterns (input $^px_n$, teacher
$^p\hat{y}_m$) from a file `training_data.txt` with up to $P = 1000$ patterns.
Use the sum over quadratic differences as error function. Allow to set
different learning rates $\eta$ for different layers.

Calculate the error in every training step, and print it during the training
process as a learning curve into a file `learning_curve.txt` and visualize it.

End the training if a predefined number of training steps has been performed,
and then test the performance of the network (no further weight changes) with
respect to a second set of data, the test set `test_data.txt` (same file format
as the training data). Choose reasonable test data on your own.

The `training_data.txt` file starts with two lines of header, followed by $P$
lines of data. Each header line starts with a `#` character followed by some
characters and strings that you can ignore (if you want to).

Each of the $P$ data lines contains the data for one pattern $p$: N-input
values $^px_1, \ldots, ^px_N$, separated by one or more blanks, M-teacher
values $^p\hat{y}_1, \ldots, ^p\hat{y}_M$, separated by blanks.

**Extra:** (no extra points) If you are experienced with neural networks and
MLPs, you can implement and train an MLP with Rectified Linear Units (ReLUs)
using the _ramp function_.

In addition, make sure that your program is running correctly, is producing the
required results, and that your source code contains valid, and useful
comments.
