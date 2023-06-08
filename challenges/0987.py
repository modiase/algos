"""
Given an array of numbers and an index i, return the index of the nearest larger number of the number at index i, where distance is measured in array indices.

For example, given [4, 1, 3, 5, 6] and index 0, you should return 3.

If two distances to larger numbers are the equal, then return any one of them. If the array at i doesn't have a nearest larger integer, then return null.

Follow-up: If you can preprocess the array, can you do this in constant time?
"""


def d_to_l(a, i):
    xs = list(enumerate(a))
    xs = sorted(xs, key=lambda t: t[1])
    sorted_i = [t[0] for t in xs].index(i)
    if sorted_i == len(a) - 1:
        return None
    seq = [abs(xs[sorted_i][0] - xs[q][0]) for q in range(sorted_i+1, len(xs))]
    if len(seq) == 0:
        return None
    return min(seq)


assert (d_to_l([4, 1, 3, 5, 6], 0)) == 3
assert (d_to_l([4, 1, 3, 5, 6], 1)) == 1
assert (d_to_l([4, 1, 3, 5, 6], 2)) == 1
assert (d_to_l([4, 1, 3, 5, 6], 3)) == 1
assert (d_to_l([4, 1, 3, 5, 6], 4)) == None
