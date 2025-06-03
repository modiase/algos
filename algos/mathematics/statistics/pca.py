from typing import Final

import numpy as np
from matplotlib import pyplot as plt

SEED: Final[int] = 42


def make_dataset() -> np.ndarray:
    """
    Generate a dataset of 100 points in 2D space.
    """
    np.random.seed(SEED)
    Xs = np.random.rand(100)
    Ys = -(Xs**2) - 2 * Xs + 1 + np.random.randn(100)
    return np.array([Xs, Ys]).T


def pca(Ds: np.ndarray) -> None:
    Ds_centered = Ds - Ds.mean(axis=0)
    Cov_XY = np.cov(Ds_centered, rowvar=False)

    for i, (eig_val, eig_vec) in enumerate(
        sorted(zip(*np.linalg.eigh(Cov_XY)), key=lambda x: x[0], reverse=True)
    ):
        plt.arrow(
            0,
            0,
            *(eig_vec * np.sqrt(eig_val)),
            head_width=0.1,
            head_length=0.1,
            fc="b",
            ec="b",
            label=f"PC{i + 1} (λ={eig_val:.2f})",
        )

    plt.scatter(Ds_centered[:, 0], Ds_centered[:, 1], alpha=0.5, label="Data points")
    plt.legend()
    plt.grid(True)
    plt.xlim(-2, 2)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("PCA Analysis")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


if __name__ == "__main__":
    pca(make_dataset())
