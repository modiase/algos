"""
Given an array of strictly the characters 'R', 'G', and 'B', segregate the
values of the array so that all the Rs come first, the Gs come second, and the
Bs come last. You can only swap elements of the array.

Do this in linear time and in-place.
"""

from collections.abc import MutableSequence


def solution(arr: MutableSequence[str]) -> None:
    if len(arr) < 3:
        return

    symbols = tuple(set(arr))
    if len(symbols) != 3:
        raise ValueError("Array must contain exactly 3 symbols")

    high_symbol = symbols[2]
    mid_symbol = symbols[1]

    low = 0
    mid = 0
    high = len(arr) - 1

    while mid <= high:
        if arr[mid] == high_symbol:
            arr[mid], arr[high] = arr[high], arr[mid]
            high -= 1
        elif arr[mid] == mid_symbol:
            mid += 1
        else:
            arr[mid], arr[low] = arr[low], arr[mid]
            low += 1


if __name__ == "__main__":
    xs = ["G", "B", "R", "R", "B", "R", "G"]
    solution(xs)
    print(xs)

    ys = [1, 0, 1, 0, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0]
    solution(ys)
    print(ys)
