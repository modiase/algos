

def find_smallest(l):
    if list(sorted(l)) == l:
        return l[0]
    idx = len(l) - 2
    jmp = len(l) // 2
    while not ((l[idx-1] > l[idx] and l[idx] < l[idx-1])):
        print(l[idx-1])
        print(l[idx])
        print(l[idx+1])
        idx = idx - jmp
        jmp = jmp // 2
        if jmp == 1:
            return l[idx+1]
    return l[idx]


assert find_smallest([5, 7, 10, 3, 4]) == 3
assert find_smallest([7, 10, 3, 4, 5]) == 3
assert find_smallest([3, 4, 5, 7, 10]) == 3
assert find_smallest([5, 3, 4]) == 3
