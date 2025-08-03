"""
This problem was asked by Facebook.

A builder is looking to build a row of N houses that can be of K different
colors. He has a goal of minimizing cost while ensuring that no two neighboring
houses are of the same color.

Given an N by K matrix where the nth row and kth column represents the cost to
build the nth house with kth color, return the minimum cost which achieves this
goal.
"""


# Original solution (unoptimized)
# def calc(vec, m):
#     return sum([m[i][vec[i][0]] for i in range(0, len(m))])


# def lexplore(i, vec, a, m):
#     col_index = vec[i][0]
#     next_index = col_index+1
#     if next_index >= len(a[i]):
#         return vec
#     new_vec = vec[0:i] + [a[i][next_index]] + vec[i+1:len(vec)]
#     for i in range(0, len(new_vec)-1):
#         if new_vec[i][0] == new_vec[i+1][0]:
#             l = lexplore(i, new_vec, a, m)
#             r = rexplore(i, new_vec, a, m)
#             if calc(l, m) <= calc(r, m):
#                 new_vec = l
#             else:
#                 new_vec = r
#     return new_vec


# def rexplore(i, vec, a, m):
#     col_index = vec[i+1][0]
#     next_index = col_index+1
#     if next_index >= len(a[i+1]):
#         return vec
#     new_vec = vec[0:i+1] + [a[i+1][next_index]] + vec[i+2:len(vec)]
#     for i in range(0, len(new_vec)-1):
#         if new_vec[i][0] == new_vec[i+1][0]:
#             l = lexplore(i, new_vec, a, m)
#             r = rexplore(i, new_vec, a, m)
#             if calc(l, m) <= calc(r, m):
#                 new_vec = l
#             else:
#                 new_vec = r
#     return new_vec


# def compute_lowest(m):
#     a = [sorted(list(enumerate(row, 0)),
#                 key=lambda x: x[1]) for row in m]
#     vec = [row[0] for row in a]
#     for i in range(0, len(vec)-1):
#         if vec[i][0] == vec[i+1][0]:
#             l = lexplore(i, vec, a, m)
#             r = rexplore(i, vec, a, m)
#             if calc(l, m) <= calc(r, m):
#                 vec = l
#             else:
#                 vec = r
#     print(m, " ", [v[0] for v in vec], " ",  calc(vec, m))


# DP solution (optimized)
def solve(m):
    n = len(m)  # number of houses
    k = len(m[0])  # number of colors
    dp = [
        [0] * k for _ in range(n)
    ]  # dp[i][j] = minimum cost to build the ith house with the jth color
    for i in range(n):
        for j in range(k):
            dp[i][j] = m[i][j]
            if i > 0:
                dp[i][j] += min(dp[i - 1][:j] + dp[i - 1][j + 1 :])
    return min(dp[-1])


# Example cases
# Test case 1: Simple case with clear minimal path
m1 = [
    [1, 2, 3, 4, 5],  # House 1
    [5, 1, 2, 3, 4],  # House 2
    [4, 5, 1, 2, 3],  # House 3
]

# Test case 2: Multiple houses have same minimal color cost
m2 = [[5, 1, 5, 5, 5], [5, 1, 5, 5, 5], [5, 5, 1, 5, 5]]

# Test case 3: All costs are equal
m3 = [[2, 2, 2, 2, 2], [2, 2, 2, 2, 2], [2, 2, 2, 2, 2]]

# Test case 4: Large cost differences
m4 = [[1, 100, 100, 100, 100], [100, 1, 100, 100, 100], [100, 100, 1, 100, 100]]

# Test case 5: Random but reasonable costs
m5 = [[4, 7, 3, 6, 2], [2, 5, 3, 4, 7], [3, 4, 6, 2, 5]]

print(solve(m1))
print(solve(m2))
print(solve(m3))
print(solve(m4))
print(solve(m5))
