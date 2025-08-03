from typing import List, Tuple

Interval = Tuple[int, int]


def is_overlapping(first_interval: Interval, second_interval: Interval) -> bool:
    return (
        first_interval[1] > second_interval[0]
        and first_interval[1] < second_interval[1]
    ) or (
        second_interval[1] < first_interval[1]
        and first_interval[0] < second_interval[1]
    )


def find_number_of_overlaps(interval: Interval, intervals: List[Interval]) -> int:
    overlaps = 0
    for comparison_interval in intervals:
        if is_overlapping(interval, comparison_interval):
            overlaps += 1
    return overlaps


def main(intervals: List[Interval]):
    overlaps_for_each_interval = [
        find_number_of_overlaps(interval, intervals) for interval in intervals
    ]

    return max(overlaps_for_each_interval)
