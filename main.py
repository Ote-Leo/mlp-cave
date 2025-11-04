import argparse as ap
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import NoReturn

from numpy.typing import NDArray

import mlp

__version__ = "0.1.0"

DEFAULT_TARNING_DATA_PATH = Path("training_data.txt")
DEFAULT_TARNING_ERROR_PATH = Path("learning_curve.txt")
DEFAULT_MODEL_PATH = Path("model.mlp")
DEFAULT_TEST_DATA_PATH = Path("test_data.txt")
DEFAULT_SHAPE = (16, 16)

DEFAULT_LOADER = "basic-raw"
DEFAULT_LOADERS = {
    "basic": mlp.load_basic,
    "basic-raw": mlp.load_basic_raw,
    "mnsit": mlp.load_mnist,
    "mnsit-reversed": mlp.load_mnist_reversed,
}


PARSER = ap.ArgumentParser(
    description=(
        "A minimalistic utility for training, testing and visualizing "
        "Multi-Layer Perceptrons (MLPs)."
    ),
)

PARSER.add_argument(
    "-v",
    "--version",
    action="store_true",
    help="print the current executable version",
)

MODEL_PARSER_GROUP = PARSER.add_subparsers(dest="subparser_name")

TRAIN_PARSER = MODEL_PARSER_GROUP.add_parser(
    "train", help="Generate an MLP trained on the input data file."
)
TRAIN_PARSER.add_argument(
    "-i",
    "--input",
    default=DEFAULT_TARNING_DATA_PATH,
    type=Path,
    help=f"path of the training data. (defaults to {DEFAULT_TARNING_DATA_PATH})",
)
TRAIN_PARSER.add_argument(
    "-o",
    "--output",
    default=DEFAULT_TARNING_ERROR_PATH,
    type=Path,
    help=f"path to record the training errors. (defaults to {DEFAULT_TARNING_ERROR_PATH})",
)
TRAIN_PARSER.add_argument(
    "-m",
    "--model-path",
    default=DEFAULT_MODEL_PATH,
    type=Path,
    help=f"path to save the trained model. (defaults to {DEFAULT_MODEL_PATH})",
)
TRAIN_PARSER.add_argument(
    "-s",
    "--shape",
    nargs="+",
    required=True,
    help="set neural network shape.",
)
TRAIN_PARSER.add_argument(
    "-l",
    "--loader",
    type=str,
    choices=tuple(DEFAULT_LOADERS),
    default=DEFAULT_LOADER,
    help=f"set dataloader. (defaults to {DEFAULT_LOADER})",
)

TEST_PARSER = MODEL_PARSER_GROUP.add_parser(
    "test",
    help=(
        "Calculate the mean square error of a given MLP with respect to a given "
        "testing data."
    ),
)
TEST_PARSER.add_argument(
    "-i",
    "--input",
    default=DEFAULT_TEST_DATA_PATH,
    type=Path,
    help=f"path of the testing data. (defaults to {DEFAULT_TEST_DATA_PATH})",
)
TEST_PARSER.add_argument(
    "-m",
    "--model-path",
    default=DEFAULT_MODEL_PATH,
    type=Path,
    help=f"path of MLP to test. (defaults to {DEFAULT_MODEL_PATH})",
)

VISUALIZE_PARSER = MODEL_PARSER_GROUP.add_parser(
    "visualize",
    help="Visualize the progression of an MLP training results.",
)
VISUALIZE_PARSER.add_argument(
    "-i",
    "--input",
    default=DEFAULT_TARNING_ERROR_PATH,
    type=Path,
    help=f"path of recorded training errors. (defaults to {DEFAULT_TARNING_ERROR_PATH})",
)
VISUALIZE_PARSER.add_argument(
    "-o",
    "--output",
    type=Path,
    required=False,
    help="path to save figure visualization.",
)


def print_version(exit_code: int = 0) -> NoReturn:
    print(__version__)
    exit(exit_code)


def load_raw(shape: Sequence[int], patterns_path: Path) -> tuple[NDArray, NDArray]:
    data = mlp.load_basic_raw(patterns_path)
    patterns_shape = shape[0]
    patterns = data[:, :patterns_shape]
    labels = data[:, patterns_shape:]
    return patterns, labels


def train(
    patterns_path: Path,
    output: Path,
    shape: Sequence[int],
    loader: Callable[[str | Path], tuple[NDArray, NDArray]],
    model_path: Path,
):
    network = mlp.Network(shape)

    if loader is mlp.load_basic_raw:
        patterns, labels = load_raw(shape, patterns_path)
    else:
        try:
            patterns, labels = loader(patterns_path)
        except ValueError:
            patterns, labels = load_raw(shape, patterns_path)

    log_file = open(output, "w+")
    mlp.train(
        network=network,
        patterns=patterns,
        labels=labels,
        error_file=log_file,
    )
    network.dump(model_path)


def test(
    test_data: Path,
    network_path: Path,
):
    network = mlp.Network.load(network_path)
    patterns, labels = load_raw(network.shape, test_data)
    print(mlp.loss(network, patterns, labels))


def visualize(errors_file: Path, fig_path: Path | None = None):
    import matplotlib.pyplot as plt

    errors = []
    with open(errors_file, "r") as file:
        for line in filter(None, map(str.strip, file)):
            errors.append(float(line))

    fig, ax = plt.subplots()
    ax.plot(errors)
    ax.set_title("Learning Curve")
    ax.set_ylabel("Mean Square Error")
    ax.set_xlabel("Learning Iteration")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    ax.grid(True, linestyle="--", alpha=0.6)
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    if fig_path:
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")

    plt.show()


def main(args: Sequence[str] | None = None) -> int:
    ns = PARSER.parse_args(args)
    if ns.version:
        print_version()

    if ns.subparser_name == "train":
        shape = tuple(map(int, ns.shape))
        train(
            ns.input,
            ns.output,
            shape,
            DEFAULT_LOADERS[ns.loader],
            ns.model_path,
        )
    elif ns.subparser_name == "test":
        test(ns.input, ns.model_path)
    else:
        visualize(ns.input, ns.output)

    return 0


if __name__ == "__main__":
    exit(main(sys.argv[1:]))
