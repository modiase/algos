def edit_distance(a: str, b: str) -> int:
    M, N = len(a), len(b)
    if M < N:
        return edit_distance(a, b)
    dp = [[0] * (N + 1) for _ in range(M + 1)]
    for i in range(N + 1):
        dp[i][0] = i
        dp[0][i] = i
    for i in range(N + 1, M + 1):
        dp[i][0] = i

    for row in dp:
        print(row)

    print()
    for i in range(1, M + 1):
        for j in range(1, N + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    for row in dp:
        print(row)
    result = dp[M][N]
    print(result)
    return result


def one_away(a: str, b: str) -> bool:
    return edit_distance(a, b) < 2


assert one_away("pale", "ple")
assert one_away("pales", "pale")
assert one_away("pale", "bale")
assert not one_away("pale", "bake")
