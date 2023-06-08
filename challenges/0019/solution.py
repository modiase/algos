

m0 = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
m1 = [[1, 2, 3], [2, 1, 3], [2, 1, 3]]
m2 = [[1, 2, 3], [3, 2, 1], [1, 2, 3]]
m3 = [[1, 2, 3], [1, 1, 2], [2, 2, 3]]
m4 = [[1, 2, 3], [1, 1, 2], [2, 2, 3], [1, 2, 3], [1, 1, 2], [3, 3, 1]]
m4 = [[1, 2, 3], [1, 1, 2], [2, 2, 3], [1, 2, 3], [1, 1, 2], [3, 3, 1]]


def calc(vec, m):
    return sum([m[i][vec[i][0]] for i in range(0, len(m))])


def lexplore(i, vec, a, m):
    col_index = vec[i][0]
    next_index = col_index+1
    if next_index >= len(a[i]):
        return vec
    new_vec = vec[0:i] + [a[i][next_index]] + vec[i+1:len(vec)]
    for i in range(0, len(new_vec)-1):
        if new_vec[i][0] == new_vec[i+1][0]:
            l = lexplore(i, new_vec, a, m)
            r = rexplore(i, new_vec, a, m)
            if calc(l, m) <= calc(r, m):
                new_vec = l
            else:
                new_vec = r
    return new_vec


def rexplore(i, vec, a, m):
    col_index = vec[i+1][0]
    next_index = col_index+1
    if next_index >= len(a[i+1]):
        return vec
    new_vec = vec[0:i+1] + [a[i+1][next_index]] + vec[i+2:len(vec)]
    for i in range(0, len(new_vec)-1):
        if new_vec[i][0] == new_vec[i+1][0]:
            l = lexplore(i, new_vec, a, m)
            r = rexplore(i, new_vec, a, m)
            if calc(l, m) <= calc(r, m):
                new_vec = l
            else:
                new_vec = r
    return new_vec


def compute_lowest(m):
    a = [sorted(list(enumerate(row, 0)),
                key=lambda x: x[1]) for row in m]
    vec = [row[0] for row in a]
    for i in range(0, len(vec)-1):
        if vec[i][0] == vec[i+1][0]:
            l = lexplore(i, vec, a, m)
            r = rexplore(i, vec, a, m)
            if calc(l, m) <= calc(r, m):
                vec = l
            else:
                vec = r
    print(m, " ", [v[0] for v in vec], " ",  calc(vec, m))


compute_lowest(m0)
compute_lowest(m1)
compute_lowest(m2)
compute_lowest(m3)
compute_lowest(m4)
