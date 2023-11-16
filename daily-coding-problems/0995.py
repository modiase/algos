"""
Given an array of integers, return a new array such that each element at index i
of the new array is the product of all the numbers in the original array except
the one at i. For example, if our input was [1, 2, 3, 4, 5], the expected output
would be [120, 60, 40, 30, 24]. If our input was [3, 2, 1], the expected output
would be [2, 3, 6].

Follow-up: what if you can't use division?
"""


def prod_except(l):
    result = []
    for i, _ in enumerate(l, 0):
        s = 1
        for j, v2 in enumerate(l, 0):
            if i == j:
                continue
            s *= v2
        result.append(s)
    return result


assert prod_except([1, 2, 3, 4, 5]) == [120, 60, 40, 30, 24]
assert prod_except([3, 2, 1]) == [2, 3, 6]
