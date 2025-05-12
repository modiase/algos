"""
Suppose we wish not only to increment a counter but also to reset it to zero (i.e.,
make all bits in it 0). Counting the time to examine or modify a bit as ‚.1/,
show how to implement a counter as an array of bits so that any sequence of n
INCREMENT and RESET operations takes time O.n/ on an initially zero counter.
(Hint: Keep a pointer to the high-order 1.)
"""

from collections.abc import MutableSequence


def increment(arr: MutableSequence[int]) -> int:
    i = 0
    while arr[i] == 1:
        arr[i] = 0
        i += 1
    arr[i] = 1
    return i


def reset(arr: MutableSequence[int], high_bit: int) -> int:
    i = high_bit
    while i >= 0:
        arr[i] = 0
        i -= 1
    return 0


if __name__ == "__main__":
    arr = [0] * 10
    hi = 0
    for i in range(1023):
        hi = max(hi, increment(arr))
    assert arr == [1] * 10
    hi = reset(arr, hi)
    assert arr == [0] * 10
