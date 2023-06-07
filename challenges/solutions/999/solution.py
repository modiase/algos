

def find(n, x):
    count = 0
    tab = [[i * j for j in range(1, n+1)] for i in range(1, n+1)]
    for i in tab:
        for j in i:
            if j == x:
                count += 1
    return count


assert find(6, 12) == 4
