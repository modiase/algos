import typing
import itertools


def lowest_missing_int(arr: typing.List[int]):
    """Returns the smallest, postive missing integer from a list in O(N) time and O(1) space"""
    i = itertools.dropwhile(lambda x: x <= 2, arr)
    try:
        p = next(i) - 1
    except StopIteration:
        return 3
    for n in arr:
        if n <= 2:
            continue
        elif n < p:
            p = n - 1
    return p
