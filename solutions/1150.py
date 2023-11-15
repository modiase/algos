"""
The skyline of a city is composed of several buildings of various widths and heights,
possibly overlapping one another when viewed from a distance. We can represent the 
buildings using an array of (left, right, height) tuples, which tell us where on 
an imaginary x-axis a building begins and ends, and how tall it is. The skyline itself 
can be described by a list of (x, height) tuples, giving the locations at which the 
height visible to a distant observer changes, and each new height.

Given an array of buildings as described above, create a function that returns the skyline.

For example, suppose the input consists of the buildings [(0, 15, 3), (4, 11, 5), 
(19, 23, 4)]. In aggregate, these buildings would create a skyline that looks like the one below.

     ______  
    |      |        ___
 ___|      |___    |   | 
|   |   B  |   |   | C |
| A |      | A |   |   |
|   |      |   |   |   |
------------------------
As a result, your function should return [(0, 3), (4, 5), (11, 3), (15, 0), (19, 4), (23, 0)].
"""
from collections import namedtuple
from typing import List, Sequence, Tuple
from heapq import heappush, heappop

Building = namedtuple('Building', ['start', 'end', 'height'])

def handle_building_start(building, tallest_building_pq, transitions):
    current_x = building.start

    if len(tallest_building_pq) != 0:
        tallest_building_y = tallest_building_pq[0][1].height
        if tallest_building_y < building.height:
            transitions.append((current_x, building.height))
    else:
        transitions.append((current_x, building.height))

    heappush(tallest_building_pq, (-building.height, building))


def handle_building_end(building, tallest_building_pq, transitions):
    current_x = building.end

    if building == tallest_building_pq[0]:
        heappop(tallest_building_pq)

        if len(tallest_building_pq) == 0:
            transitions.append((current_x, 0))
        else:
            new_tallest_building = tallest_building_pq[0][1]
            transitions.append((current_x, new_tallest_building.height))
    else:
        print(tallest_building_pq)
        tallest_building_pq = [ (-x[1].height, x[1]) for x in tallest_building_pq if x[1] != building ] # O(N) Can we do better?
        print(tallest_building_pq)


def solution(buildings: Sequence[Tuple[int, int, int]]) -> List[Tuple[int, int]]:
    if len(buildings) == 0:
        return []

    sorted_buildings = [ Building(start=x[0], end=x[1], height=x[2]) for x in sorted(buildings, key=lambda t: t[0])]

    tallest_building_pq = []


    transitions = []
    events = []

    for building in sorted_buildings:
        heappush(events, (building.start, 'start', building))
        heappush(events, (building.end, 'end', building))

    for event in events:
        if event[1] == 'start':
            handle_building_start(event[2], tallest_building_pq, transitions)
        if event[1] == 'end':
            handle_building_end(event[2], tallest_building_pq, transitions)


    return transitions



test_input = [(0, 15, 3), (4, 11, 5), (19, 23, 4)]
print(solution(test_input))

