from collections.abc import MutableSequence
from itertools import batched


def solution(xs: MutableSequence[int]) -> None:
    """
    Solution is probably not optimal.
    Worst case O(n^2) which makes it pointless since we could achieve the same
    thing in O(n) using counting sort or O(n log n) using any optimized
    comparison sort.
    """
    N = len(xs)
    symbols = tuple(set(xs))
    if len(symbols) == len(xs):
        return

    high = N - 1
    for pair in batched(symbols, 2):
        if len(pair) == 1:
            break
        medium_symbol, high_symbol = pair

        low = 0
        mid = 0

        while mid <= high:
            if xs[mid] == high_symbol:
                xs[mid], xs[high] = xs[high], xs[mid]
                high -= 1
            elif xs[mid] == medium_symbol:
                mid += 1
            else:
                xs[mid], xs[low] = xs[low], xs[mid]
                low += 1
                mid += 1
        high = max(low - 1, 0)


if __name__ == "__main__":
    # xs = [1, 0, 1, 0, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0]
    # solution(xs)
    # print(xs)

    xs = [2, 2, 3, 5, 5, 5, 1, 1, 4, 4, 5, 5, 5, 3, 3, 5, 1, 2, 2, 5]
    solution(xs)
    print(xs)
