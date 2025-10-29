def find_smallest(arr):
    if list(sorted(arr)) == arr:
        return arr[0]
    idx = len(arr) - 2
    jmp = len(arr) // 2
    while not (arr[idx - 1] > arr[idx] and arr[idx] < arr[idx - 1]):
        print(arr[idx - 1])
        print(arr[idx])
        print(arr[idx + 1])
        idx = idx - jmp
        jmp = jmp // 2
        if jmp == 1:
            return arr[idx + 1]
    return arr[idx]


assert find_smallest([5, 7, 10, 3, 4]) == 3
assert find_smallest([7, 10, 3, 4, 5]) == 3
assert find_smallest([3, 4, 5, 7, 10]) == 3
assert find_smallest([5, 3, 4]) == 3
