import math


def fibonacci_binet(n: int) -> int:
    """
    Compute the nth Fibonacci number using Binet's formula.

    Args:
        n (int): The position in the Fibonacci sequence (0-based)

    Returns:
        int: The nth Fibonacci number

    Raises:
        ValueError: If n is negative
    """
    if n < 0:
        raise ValueError("Fibonacci sequence not defined for negative numbers")

    # Calculate the golden ratio (phi)
    phi = (1 + math.sqrt(5)) / 2

    # Calculate Binet's formula: (phi^n - (-phi)^(-n))/sqrt(5)
    return round((phi**n - (-phi) ** (-n)) / math.sqrt(5))


# Test the function
if __name__ == "__main__":
    # Test first 10 Fibonacci numbers
    print("First 10 Fibonacci numbers:")
    for i in range(10):
        print(f"F({i}) = {fibonacci_binet(i)}")

    # Verify with larger numbers
    print("\nSome larger Fibonacci numbers:")
    for i in [20, 30, 40]:
        print(f"F({i}) = {fibonacci_binet(i)}")
