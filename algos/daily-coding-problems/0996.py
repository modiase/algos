"""
Recall that the minimum spanning tree is the subset of edges of a tree that
connect all its vertices with the smallest possible total edge weight. Given
an undirected graph with weighted edges, compute the maximum weight spanning
tree.
"""

import itertools

g1 = [
    [0, 9, 6, -1, 9, -1],
    [9, 0, 3, 5, -1, -1],
    [6, 3, 0, -1, 4, 12],
    [-1, 5, -1, 0, -1, 10],
    [9, -1, 4, -1, 0, 7],
    [-1, -1, 12, 10, 7, 0],
]


def rm_edge(graph, m, n):
    new_graph = [[*vertex] for vertex in graph]
    new_graph[m][n] = -1
    new_graph[n][m] = -1
    return new_graph


def get_edges(g):
    edges = []
    for vertex, weights in enumerate(g):
        for neighbour, weight in enumerate(weights, 0):
            if weight not in [-1, 0] and (neighbour, vertex, weight) not in edges:
                edges.append((vertex, neighbour, weight))
    return edges


def ordered_edges(edges):
    return sorted(edges, key=lambda t: t[2], reverse=True)


def gsum(g) -> int:
    sum = 0
    for vertex in g:
        for edge in vertex:
            if edge != -1:
                sum += edge
    return sum


def ecount(g):
    count = 0
    for vertex in g:
        for edge in vertex:
            if edge != -1:
                count += 1
    return count


def is_acyclic(g):
    def _is_acyclic(g, v, visited):
        neighbours = [
            neighbour
            for neighbour, weight in enumerate(g[v], 0)
            if weight not in [0, -1]
        ]
        if len(neighbours) == 0:
            return None
        for neighbour in neighbours:
            if neighbour in visited:
                return [*visited, neighbour]
        paths = [_is_acyclic(rm_edge(g, v, n), n, [*visited, v]) for n in neighbours]
        for path in paths:
            if path is not None:
                return path

    return _is_acyclic(g, 0, [])


def is_connected(g):
    for vertex in g:
        if all([edge in [0, -1] for edge in vertex]):
            return False
    return True


def is_spanning_tree(g):
    return is_connected(g) and is_acyclic(g) is None


def copy_graph(g):
    return [[*vertex] for vertex in g]


maxg = None
maxweight = 0

ming = None
minweight = float("inf")

for i in range(1, ecount(g1)):
    pairs = itertools.combinations(range(0, len(g1)), 2)
    combinations = itertools.combinations(pairs, i)
    for combination in combinations:
        cp = copy_graph(g1)
        for edge in combination:
            cp = rm_edge(cp, edge[0], edge[1])
        if is_spanning_tree(cp) and gsum(cp) > maxweight:
            maxg = cp
            maxweight = gsum(cp)
        if is_spanning_tree(cp) and gsum(cp) < minweight:
            ming = cp
            minweight = gsum(cp)

print(maxweight)
print(maxg)
print(minweight)
print(ming)
