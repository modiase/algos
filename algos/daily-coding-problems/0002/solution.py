from functools import reduce


def reducer(l, index):
    prod = 1
    for i in range(0, len(l)):
        if index == i:
            continue
        prod *= l[i]
    return prod


def product_except_at_index(l):
    result = [reducer(l, i) for i in range(0, len(l))]
    return result


if __name__ == "__main__":
    t_array = [1, 2, 3, 4, 5]
    result = product_except_at_index(t_array)
    assert result == [120, 60, 40, 30, 24]
