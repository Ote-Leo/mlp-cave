import sys
from collections.abc import Sequence


def main(args: Sequence[str] | None = None) -> int:
    print("Hello, from mlp-cave")
    return 0


if __name__ == "__main__":
    exit(main(sys.argv[1:]))
