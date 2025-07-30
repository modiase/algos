#!/usr/bin/env python3
import os
from datetime import datetime

import jax
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from more_itertools import one


def main() -> None:
    seed = int(os.environ.get("SEED", datetime.now().timestamp()))
    logger.info(f"Using {seed=}")
    key = jrandom.PRNGKey(seed)

    key, subkey = jrandom.split(key)
    points = jrandom.uniform(subkey, (10, 3), minval=-5, maxval=5)

    key, subkey = jrandom.split(key)
    random_vector = jrandom.normal(subkey, (3,))
    v = random_vector / jnp.linalg.norm(random_vector)
    logger.info(f"Random unit vector: {v}")

    H = jnp.eye(one(v.shape)) - 2 * jnp.einsum("i,j->ij", v, v)
    logger.info(f"Householder reflection matrix: \n{H=}")

    reflected_points = jax.vmap(lambda point: H @ point)(points)

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c="blue",
        s=100,
        alpha=0.7,
        label="Original points",
    )

    ax.scatter(
        reflected_points[:, 0],
        reflected_points[:, 1],
        reflected_points[:, 2],
        c="green",
        s=100,
        alpha=0.7,
        label="Reflected points",
    )

    # Plot the Householder vector in purple
    origin = jnp.array([0, 0, 0])
    ax.quiver(
        *origin,
        *3 * v,
        color="purple",
        linewidth=3,
        arrow_length_ratio=0.1,
        label="Householder vector",
    )

    xx, yy = np.meshgrid(np.linspace(-6, 6, 10), np.linspace(-6, 6, 10))

    if abs(v[2]) > 1e-6:
        zz = -(v[0] * xx + v[1] * yy) / v[2]
        ax.plot_surface(xx, yy, zz, alpha=0.3, color="yellow", label="Reflection plane")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Householder Reflection")
    ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
