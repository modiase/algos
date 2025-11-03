def recursive_factorial(n: int) -> int:
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    if n == 0:
        return 1
    return n * recursive_factorial(n - 1)


def iterative_factorial(n: int) -> int:
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    stack = [(1, 0)]
    result = 1
    while stack:
        result, i = stack.pop()
        if i != n:
            stack.append((result * (i + 1), i + 1))
    return result


if __name__ == "__main__":
    assert iterative_factorial(0) == recursive_factorial(0) == 1
    assert iterative_factorial(1) == recursive_factorial(1) == 1
    assert iterative_factorial(2) == recursive_factorial(2) == 2
    assert iterative_factorial(3) == recursive_factorial(3) == 6
    assert iterative_factorial(4) == recursive_factorial(4) == 24
    assert iterative_factorial(5) == recursive_factorial(5) == 120
