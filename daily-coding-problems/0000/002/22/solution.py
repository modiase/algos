from typing import Sequence, Tuple

TInterval = Tuple[int, int]


def count_overlaps(index: int, interval: TInterval, list_of_intervals: Sequence[TInterval]) -> int:
    return sum([(interval[0] > comparison_interval[1] or interval[1] < comparison_interval[0])
                for idx, comparison_interval in enumerate(list_of_intervals) if index != idx])


def find_maximum_overlap(list_of_intervals: Sequence[TInterval]) -> int:
    max_overlaps = max([count_overlaps(index, interval, list_of_intervals)
                        for index, interval in enumerate(list_of_intervals)])
    return max_overlaps


def main(list_of_intervals: Sequence[TInterval]) -> int:
    required_rooms = find_maximum_overlap(list_of_intervals) + 1
    return required_rooms
