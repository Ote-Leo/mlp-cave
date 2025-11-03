import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from numpy.typing import NDArray

import mlp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

IMAGE_WIDTH: int = 256
IMAGE_HEIGHT: int = IMAGE_WIDTH

MODEL_PATH = "models/mnist-reversed/mnist-reversed.mlp"
network = mlp.Network.load(MODEL_PATH)


def compute_layer(t: np.float64) -> tuple[np.float64, NDArray]:
    print(f"rendering frame at timestep {t:.2f}")
    frame = np.empty((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float64)
    for y in range(IMAGE_HEIGHT):
        for x in range(IMAGE_WIDTH):
            frame[y, x] = network.forward(
                np.array([x / IMAGE_WIDTH, y / IMAGE_HEIGHT, t])
            )[0]
    return t, frame


if __name__ == "__main__":
    timesteps = np.arange(0, 9.1, 0.01)
    ys, xs = np.linspace(0, 1, IMAGE_HEIGHT), np.linspace(0, 1, IMAGE_WIDTH)
    frames = np.empty((len(timesteps), IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float64)

    with ProcessPoolExecutor(max_workers=50) as exec:
        future_to_index = {
            exec.submit(compute_layer, t): i  # type: ignore
            for i, t in enumerate(timesteps)
        }

        for future in as_completed(future_to_index):
            i = future_to_index[future]
            t, frame = future.result()
            print(f"finished rendering frame at timestep {t:.2f}")
            frames[i] = frame

    fig, ax = plt.subplots()

    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.patch.set_alpha(0)

    im = ax.imshow(frames[0], cmap="gray", vmin=0, vmax=1)

    def update(frame_index):
        im.set_data(frames[frame_index])
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(frames), blit=True)
    gif_path = "assets/animation.gif"
    ani.save(gif_path, writer="pillow", fps=20, savefig_kwargs={"transparent": True})
    plt.show()
    plt.close(fig)
    print(f"saved gif to {gif_path}")
