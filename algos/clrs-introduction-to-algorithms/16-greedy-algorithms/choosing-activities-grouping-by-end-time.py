from __future__ import annotations
from collections.abc import Collection
from typing import NamedTuple


class Activity(NamedTuple):
    """Activity with start and end times"""

    start: int
    end: int

    def __str__(self) -> str:
        return f"({self.start}, {self.end})"

    def __hash__(self) -> int:
        return hash((self.start, self.end)) + id(self)

    def overlaps_with(self, other: Activity) -> bool:
        return self.start < other.end and other.start < self.end


def optimal_group_activities(activities: Collection[Activity]) -> list[list[Activity]]:
    """
    Groups activities using the earliest finish time first greedy algorithm.
    """
    if not activities:
        return []

    activities_sorted_by_end_time = sorted(activities, key=lambda x: x.end)

    groups: list[list[Activity]] = []

    for activity in activities_sorted_by_end_time:
        for group in groups:
            if not any(
                activity.overlaps_with(group_activity) for group_activity in group
            ):
                group.append(activity)
                break
        else:
            groups.append([activity])

    return groups


def print_groups(groups):
    for i, group in enumerate(groups):
        activities_str = ", ".join(str(activity) for activity in group)
        print(f"Group {i + 1}: {activities_str}")
    print(f"Total groups: {len(groups)}")


if __name__ == "__main__":
    test_cases = [
        [Activity(0, 2), Activity(3, 5), Activity(6, 8)],  # No overlaps
        [Activity(3, 7), Activity(4, 8), Activity(2, 6)],  # All overlap
        [Activity(1, 4)],  # Single activity
        [Activity(1, 3), Activity(2, 4), Activity(3, 5), Activity(4, 6)],  # Chain
        [
            Activity(1, 3),
            Activity(2, 4),
            Activity(6, 8),
            Activity(7, 9),
            Activity(11, 13),
        ],  # Multiple non-overlapping groups
        [
            Activity(1, 10),
            Activity(2, 9),
            Activity(3, 8),
            Activity(4, 7),
        ],  # Nested intervals
    ]

    for i, test_case in enumerate(test_cases):
        print(f"\nTest case {i + 1}:")
        groups = optimal_group_activities(test_case)
        print_groups(groups)
