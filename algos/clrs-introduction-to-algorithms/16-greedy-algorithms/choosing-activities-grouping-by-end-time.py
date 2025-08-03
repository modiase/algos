from __future__ import annotations
from collections.abc import Collection
from typing import NamedTuple, List


class Activity(NamedTuple):
    """Activity with start and end times"""

    start: int
    end: int

    def __str__(self) -> str:
        """String representation of activity"""
        return f"({self.start}, {self.end})"

    def __hash__(self) -> int:
        """Hash for activity, used in sets and dicts"""
        return hash((self.start, self.end)) + id(self)

    def overlaps_with(self, other: Activity) -> bool:
        """Return True if activities overlap"""
        return self.start < other.end and other.start < self.end


def optimal_group_activities(activities: Collection[Activity]) -> List[List[Activity]]:
    """
    Groups activities using the earliest finish time first greedy algorithm.

    This algorithm is guaranteed to produce the minimum number of groups
    (equivalent to finding the minimum coloring of an interval graph).

    Returns a list of groups, where each group is a list of non-overlapping activities.
    """
    if not activities:
        return []

    # Sort activities by end time (earliest finish time first)
    sorted_activities = sorted(activities, key=lambda x: x.end)

    # Initialize empty groups
    groups: List[List[Activity]] = []

    # Process each activity in sorted order
    for activity in sorted_activities:
        # Try to add the activity to an existing group
        for group in groups:
            # Check if activity overlaps with any activity in this group
            if not any(
                activity.overlaps_with(group_activity) for group_activity in group
            ):
                group.append(activity)
                break
        else:
            # If no compatible group found, create a new group
            groups.append([activity])

    return groups


# Counter-example activities
counter_example = [
    Activity(1, 5),  # A
    Activity(2, 9),  # B
    Activity(6, 10),  # C
    Activity(8, 12),  # D
    Activity(11, 15),  # E
    Activity(3, 7),  # F
    Activity(4, 14),  # G
]


# Function to print results in a readable format
def print_groups(groups):
    for i, group in enumerate(groups):
        activities_str = ", ".join(str(activity) for activity in group)
        print(f"Group {i + 1}: {activities_str}")
    print(f"Total groups: {len(groups)}")


# Run the algorithm on the counter-example
if __name__ == "__main__":
    print("Activities sorted by earliest finish time:")
    for activity in sorted(counter_example, key=lambda x: x.end):
        print(f"  {activity}")
    print("\nOptimal grouping:")
    groups = optimal_group_activities(counter_example)
    print_groups(groups)

    # Original test cases from the input
    print("\nRunning with original test cases:")
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
