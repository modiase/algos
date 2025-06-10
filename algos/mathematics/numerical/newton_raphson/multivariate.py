from collections.abc import Callable
from typing import Final

import jax.numpy as jnp
from jax import grad, hessian, jit
from loguru import logger

EPSILON: Final[float] = 1e-4


@jit
def f(x: jnp.ndarray) -> float:
    """
    This function is quadratic, so the Newton-Raphson method should converge in one iteration.
    """
    _x, _y = x
    return _x**2 + _y**2


def newton_raphson_multivariate(
    f: Callable[[jnp.ndarray], float],
    x0: jnp.ndarray,
    alpha: float = 1.0,
    tol: float = 1e-6,
    max_iter: int = 1000,
):
    # Recall that we can use NR for optimization by finding the root of the gradient
    # ∇f(x) = 0
    # We can use the Newton-Raphson method to find the root of the gradient
    # by iterating the following update:
    # x_{n+1} = x_n - H⁻¹ · ∇f(x_n)

    x = x0
    df = grad(f)
    d2f = hessian(f)

    for i in range(max_iter):
        J = df(x)
        H = d2f(x)

        if (norm := jnp.linalg.norm(J)) < tol:
            logger.info(f"Newton-Raphson method converged after {i} iterations: {norm}")
            return x

        try:
            # ∇f(x + dx) ≈ ∇f(x) + H · dx
            # We seek dx such that ∇f(x) + H · dx = 0
            # => dx = -H⁻¹ · ∇f(x)
            dx = -jnp.linalg.solve(H, J)
            x += alpha * dx
        except jnp.linalg.LinAlgError:
            logger.error("Hessian matrix is singular, cannot continue")
            break

    raise ValueError(
        f"Newton-Raphson method did not converge after {max_iter} iterations: {jnp.linalg.norm(df(x))}"
    )


if __name__ == "__main__":
    x0 = jnp.array([1.0, 2.0])
    logger.info(f"Initial function value: {f(x0)}")
    logger.info(f"Result: {newton_raphson_multivariate(f, x0)}")
