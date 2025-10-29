"""
A teacher must divide a class of students into two teams to play dodgeball. Unfortunately, not all the kids get along, and several refuse to be put on the same team as that of their enemies.

Given an adjacency list of students and their enemies, write an algorithm that finds a satisfactory pair of teams, or returns False if none exists.

For example, given the following enemy graph you should return the teams {0, 1, 4, 5} and {2, 3}.
students = {
    0: [3],
    1: [2],
    2: [1, 4],
    3: [0, 4, 5],
    4: [2, 3],
    5: [3]
}
On the other hand, given the input below, you should return False.

students = {
    0: [3],
    1: [2],
    2: [1, 3, 4],
    3: [0, 2, 4, 5],
    4: [2, 3],
    5: [3]
}
"""

from typing import List, Literal, Mapping, cast


m0 = {0: [3], 1: [2], 2: [1, 4], 3: [0, 4, 5], 4: [2, 3], 5: [3]}

m1 = {0: [3], 1: [2], 2: [1, 3, 4], 3: [0, 2, 4, 5], 4: [2, 3], 5: [3]}

K = Literal[0, 1, 2, 3, 4, 5]
V = Literal[0, 1]


def solution(m: Mapping[K, List[V]]):
    team_map: Mapping[K, V] = cast(Mapping[K, V], {k: 0 for k in m})

    # TODO
    # Return false if no teams can be found
    if is_consistent(team_map):
        return (
            [k for k in team_map if team_map[k]],
            [k for k in team_map if not team_map[k]],
        )

    return False


def is_consistent(team_map: Mapping[K, V]):
    pass


assert sorted(solution(m0)) == sorted([{0, 1, 4, 5}, {2, 3}])
assert not solution(m1)
