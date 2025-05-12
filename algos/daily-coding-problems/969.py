"""
Given a matrix of 1s and 0s, return the number of "islands" in the matrix. A 1 represents land and 0 represents water, so an island is a group of 1s that are neighboring whose perimeter is surrounded by water.

For example, this matrix has 4 islands.

1 0 0 0 0
0 0 1 1 0
0 1 1 0 0
0 0 0 0 0
1 1 0 0 1
1 1 0 0 1

Solved: 22m
"""

m0 = [
    [1,0,0,0,0],
    [0,0,1,1,0],
    [0,1,1,0,0],
    [0,0,0,0,0],
    [1,1,0,0,1],
    [1,1,0,0,1],
]

def make_clusters(m, ones):
    ones = set(ones)
    clusters = {}
    clustered: dict[tuple, tuple | None] = { t: None for t in ones }

    for i0, j0 in ones:

        clusters[(i0,j0)] = [(i0,j0)]
        clustered[(i0,j0)] = (i0,j0)

        for x in range(-1,1):
            for y in range(-1, 1):
                if x == 0 and y == 0:
                    continue
                else:
                    i = i0 + x
                    j = j0 + y
                    if (i,j) not in ones:
                        continue
                    if i < 0 or i >= len(m):
                        continue
                    if j < 0 or j >= len(m[0]):
                        continue

                    if clustered[(i,j)] is not None:
                        cluster = clustered[(i,j)]
                        self_cluster_centre = clustered[(i0, j0)]
                        self_cluster = clusters[self_cluster_centre]
                        clusters[cluster].extend(self_cluster)
                        for one in self_cluster:
                            clustered[one] = cluster
                        del clusters[self_cluster_centre]
                        break
                    else:
                        cluster = clustered[(i0, j0)]
                        clusters[cluster].append((i,j))
                        clustered[(i,j)] = cluster

    return clusters

def solution(m):
    ones = []
    for i in range(len(m)):
        for j in range(len(m[0])):
            if m[i][j]:
                ones.append((i,j))



    return len(make_clusters(m, ones))



assert(solution(m0) == 4)

