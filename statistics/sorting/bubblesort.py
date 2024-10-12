from typing import Iterable


def bubblesort(iterable: Iterable[int]) -> tuple[list[int], int]:
    result = list(iterable)
    swaps = 0
    N = len(result)
    for start in range(1, N):
        for i in range(start, 0, -1):
            if result[i] < result[i - 1]:
                swaps += 1
                tmp = result[i - 1]
                result[i - 1] = result[i]
                result[i] = tmp
                continue
            break

    return result, swaps


if __name__ == "__main__":
    print(bubblesort([5, 2, 4, 1, 3]))
