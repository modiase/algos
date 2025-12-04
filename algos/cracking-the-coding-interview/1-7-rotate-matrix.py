def rotate_indices(i: int, j: int, N: int) -> tuple[int, int]:
    # [   cos t  sin t ]
    # [ - sin t  cos t ]
    # t = pi/2  => [ [0 1] [-1 0] ]
    return j, (N - 1) - i


def rotate_matrix(M: list[list[int]]) -> None:
    if not M:
        return
    N = len(M)
    if N != len(M[0]):
        raise ValueError("M must be square")
    i_start = (N // 2) + 1 if N % 2 == 1 else N // 2
    j_start = N // 2
    for i in range(i_start, N):
        for j in range(j_start, N):
            start_indices = current_indices = (i, j)
            current_value = M[i][j]
            while True:
                ni, nj = rotate_indices(*current_indices, N)
                tmp = M[ni][nj]
                M[ni][nj] = current_value
                current_value = tmp
                current_indices = (ni, nj)

                if current_indices == start_indices:
                    break


def print_matrix(M: list[list[int]]) -> None:
    for r in M:
        print(r)


# print(rotate_indices(0, 0, 3))
# print(rotate_indices(0, 2, 3))
# print(rotate_indices(2, 2, 3))
# print(rotate_indices(2, 0, 3))

# print(rotate_indices(0, 1, 3))
# print(rotate_indices(1, 2, 3))
# print(rotate_indices(2, 1, 3))
# print(rotate_indices(1, 0, 3))


for size in range(5):
    M = [[0] * size for _ in range(size)]
    for row in range(size):
        for col in range(size):
            M[row][col] = row * size + col

    print(f"{size=}")
    print_matrix(M)
    print()
    rotate_matrix(M)
    print_matrix(M)
    print()
    print()
