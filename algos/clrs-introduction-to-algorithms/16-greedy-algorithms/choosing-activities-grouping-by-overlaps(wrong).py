"""
This solution is wrong because the greedy solution by least overlaps does not
lead to a global optimal solution.  This is the case because this greedy
strategy will sometimes choose an activity that has few overlaps but locks the
solution into a suboptimal grouping.
"""

from __future__ import annotations

import heapq
from collections.abc import Mapping, Sequence
from typing import NamedTuple


class Activity(NamedTuple):
    start: int
    end: int

    def __hash__(self) -> int:
        return hash((self.start, self.end)) + id(self)

    def __or__(self, other: Activity) -> Activity | None:
        if self.start <= other.start <= self.end or self.start <= other.end <= self.end:
            return Activity(min(self.start, other.start), max(self.end, other.end))
        return None


def find_overlapping_activities(
    activities: Sequence[Activity],
) -> Mapping[Activity, set[Activity]]:
    if not activities:
        return {}

    min_time = min(activity.start for activity in activities)
    max_time = max(activity.end for activity in activities)
    range_size = max_time - min_time + 1

    overlaps = {activity: set() for activity in activities}

    # Special case: if range is too large relative to n, fall back to O(n^2) to
    # avoid excessive memory use
    if range_size > 10 * len(activities):
        for i, a1 in enumerate(activities):
            for a2 in activities[i + 1 :]:
                if a1.start < a2.end and a2.start < a1.end:
                    overlaps[a1].add(a2)
                    overlaps[a2].add(a1)
        return overlaps

    buckets = [[] for _ in range(range_size)]

    for activity in activities:
        for t in range(activity.start - min_time, activity.end - min_time):
            buckets[t].append(activity)

    for bucket in buckets:
        if len(bucket) > 1:
            for i, a1 in enumerate(bucket):
                for a2 in bucket[i + 1 :]:
                    overlaps[a1].add(a2)
                    overlaps[a2].add(a1)

    return overlaps


class PriorityQueue:
    def __init__(self, activities_with_overlaps: Mapping[Activity, set[Activity]]):
        self._heap = [
            (-len(value), (key, value))
            for key, value in activities_with_overlaps.items()
        ]
        heapq.heapify(self._heap)
        self._index = {activity: i for i, (_, (activity, _)) in enumerate(self._heap)}

    def _swap(self, i: int, j: int) -> None:
        self._heap[i], self._heap[j] = self._heap[j], self._heap[i]
        self._index[self._heap[i][1][0]], self._index[self._heap[j][1][0]] = (
            self._index[self._heap[j][1][0]],
            self._index[self._heap[i][1][0]],
        )

    def lchild(self, i: int) -> int | None:
        return 2 * i + 1 if 2 * i + 1 < len(self._heap) else None

    def rchild(self, i: int) -> int | None:
        return 2 * i + 2 if 2 * i + 2 < len(self._heap) else None

    def parent(self, i: int) -> int | None:
        return (i - 1) // 2 if i > 0 else None

    def remove_overlap(
        self, activity: Activity, overlapping_activity: Activity
    ) -> None:
        _, (_, overlapping) = self._heap[self._index[activity]]
        overlapping.remove(overlapping_activity)
        self._heap[self._index[activity]] = (-len(overlapping), (activity, overlapping))
        self._sift_down(activity)

    def _sift_down(self, activity: Activity) -> None:
        idx = self._index[activity]
        while True:
            left = self.lchild(idx)
            right = self.rchild(idx)
            smallest = idx

            if left is not None and self._heap[left][0] < self._heap[smallest][0]:
                smallest = left
            if right is not None and self._heap[right][0] < self._heap[smallest][0]:
                smallest = right

            if smallest == idx:
                break

            self._swap(idx, smallest)
            idx = smallest

    def pop(self) -> Activity | None:
        if not self._heap:
            return None
        self._swap(0, len(self._heap) - 1)
        _, (activity, overlaps) = self._heap.pop()
        for overlapping_activity in overlaps:
            self.remove_overlap(overlapping_activity, activity)
        del self._index[activity]
        if self._heap:
            self._sift_down(self._heap[0][1][0])
        return activity


def group_activities(
    activities: Sequence[Activity],
) -> list[list[Activity]]:
    queue = PriorityQueue(find_overlapping_activities(activities))
    groups: list[list[Activity]] = []
    while queue._heap:
        activity = queue.pop()
        if activity is None:
            break
        for group in groups:
            if any((activity | group_activity) is not None for group_activity in group):
                continue
            group.append(activity)
            break
        else:
            groups.append([activity])
    return groups


# Test case 1: No overlaps
test_case_1 = [Activity(0, 2), Activity(3, 5), Activity(6, 8)]

# Test case 2: All activities overlap
test_case_2 = [
    Activity(3, 7),
    Activity(4, 8),
    Activity(2, 6),
]

# Test case 3: Single activity
test_case_3 = [Activity(1, 4)]

# Test case 4: Partial overlaps forming a chain
test_case_4 = [Activity(1, 3), Activity(2, 4), Activity(3, 5), Activity(4, 6)]

# Test case 5: Multiple non-overlapping groups
test_case_5 = [
    Activity(1, 3),
    Activity(2, 4),  # overlaps with 1
    Activity(6, 8),
    Activity(7, 9),  # overlaps with 6
    Activity(11, 13),
]

# Test case 6: Nested intervals
test_case_6 = [Activity(1, 10), Activity(2, 9), Activity(3, 8), Activity(4, 7)]

# Test case 7: Edge case - activities starting at same time
test_case_7 = [Activity(5, 7), Activity(5, 8), Activity(5, 9), Activity(8, 10)]

# Test case 8: Edge case - activities ending at same time
test_case_8 = [Activity(1, 6), Activity(2, 6), Activity(3, 6), Activity(7, 9)]

# Test case 9: Mixed overlaps
test_case_9 = [
    Activity(1, 4),
    Activity(2, 5),
    Activity(6, 8),
    Activity(7, 9),
    Activity(10, 12),
    Activity(11, 13),
]

# Test case 10: Empty case
test_case_10 = []

test_cases = [
    test_case_1,
    test_case_2,
    test_case_3,
    test_case_4,
    test_case_5,
    test_case_6,
    test_case_7,
    test_case_8,
    test_case_9,
    test_case_10,
]

if __name__ == "__main__":
    for test_case in test_cases:
        groups = group_activities(test_case)
        print(groups)
