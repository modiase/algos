"""
The transitive closure of a graph is a measure of which vertices are reachable from other vertices. It can be represented as a matrix M, where M[i][j] == 1 if there is a path between vertices i and j, and otherwise 0.

For example, suppose we are given the following graph in adjacency list form:

graph = [
    [0, 1, 3],
    [1, 2],
    [2],
    [3]
]
The transitive closure of this graph would be:

[1, 1, 1, 1]
[0, 1, 1, 0]
[0, 0, 1, 0]
[0, 0, 0, 1]
Given a graph, find its transitive closure.
"""

graph = [[0, 1, 3], [1, 2], [2], [3]]


def tclosure(g):
    n = len(g)
    mat = [[0 for _ in range(n)] for _ in range(n)]
    for v in range(n):
        for m in range(v, n):
            if m in g[v]:
                mat[v][m] = 1

    for o in range(n):
        for p in range(n):
            connected = mat[o][p]
            if connected:
                for i in range(n):
                    if mat[p][i] == 1:
                        mat[o][i] = 1
    return mat


assert tclosure(graph) == [
    [1, 1, 1, 1],
    [0, 1, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
]
