from typing import Sequence


def solve(
    seq_a: str, seq_b: str
) -> tuple[Sequence[Sequence[int]], Sequence[Sequence[str]]]:
    m = len(seq_a)
    n = len(seq_b)
    c = [[0] * (n + 1) for _ in range(m + 1)]
    b = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq_a[i - 1] == seq_b[j - 1]:
                c[i][j] = c[i - 1][j - 1] + 1
                b[i][j] = "↖"
            elif c[i - 1][j] >= c[i][j - 1]:
                c[i][j] = c[i - 1][j]
                b[i][j] = "↑"
            else:
                c[i][j] = c[i][j - 1]
                b[i][j] = "←"

    return c, b


def reconstruct(
    c: Sequence[Sequence[int]], seq_a: Sequence[str], seq_b: Sequence[str]
) -> Sequence[str]:
    i = len(seq_a)
    j = len(seq_b)
    res = []
    while i > 0 and j > 0:
        if c[i][j] == c[i - 1][j - 1] + 1:
            res.append(seq_a[i - 1])
            i -= 1
            j -= 1
        elif c[i][j] == c[i - 1][j]:
            i -= 1
        else:
            j -= 1
    return res[::-1]


if __name__ == "__main__":
    seq_a = "ABCDABAB"
    seq_b = "QRZABCDQ"
    c, b = solve(seq_a, seq_b)
    for row in c:
        print(row)
    print()
    for row in b:
        print(row)

    print(reconstruct(c, seq_a, seq_b))
