from collections.abc import Callable
from math import sqrt
from typing import Final


def x_squared(x: float) -> float:
    return x**2


def some_polynomial(x: float) -> float:
    return 2 * x**2 + 10 * x + 10


def some_polynomial_2(x: float) -> float:
    return x**3 - 2 * x**2 + 5 * x + 10


def find_machine_epsilon():
    eps = 1.0
    while 1.0 + eps / 2 > 1.0:
        eps /= 2
    return eps


Func = Callable[[float], float]
EPSILON: Final[float] = find_machine_epsilon()
h = sqrt(EPSILON)
MAX_ITER: Final[int] = 1000


def deriv(f: Func) -> Func:
    return lambda x: (f(x + h) - f(x - h)) / (2 * h)


def newton_raphson(
    f: Func, x: float, tol: float = EPSILON, max_iter: int = MAX_ITER
) -> float:
    def _newton_raphson_step(f: Func, x: float) -> float:
        return x - f(x) / deriv(f)(x)

    for _ in range(max_iter):
        x = _newton_raphson_step(f, x)
        if abs(f(x)) < tol:
            return x
    raise ValueError(f"Failed to converge. Final value: {x=}")


if __name__ == "__main__":
    print(newton_raphson(x_squared, 1))
    print(newton_raphson(some_polynomial, 1))
    print(newton_raphson(some_polynomial_2, 1))
