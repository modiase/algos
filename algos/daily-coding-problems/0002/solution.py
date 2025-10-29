def reducer(arr, index):
    prod = 1
    for i in range(0, len(arr)):
        if index == i:
            continue
        prod *= arr[i]
    return prod


def product_except_at_index(arr):
    result = [reducer(arr, i) for i in range(0, len(arr))]
    return result


if __name__ == "__main__":
    t_array = [1, 2, 3, 4, 5]
    result = product_except_at_index(t_array)
    assert result == [120, 60, 40, 30, 24]
