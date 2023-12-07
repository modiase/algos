"""
"""
import path
import re
from typing import List

LINE_REGEX = re.compile(r'Card\s*(\d+):\s*((?:\d+\s*)+)\|\s*((?:\d+\s*)+)')


def parse_card_value(line: str) -> int:
    parsed = LINE_REGEX.match(line)
    if parsed is None:
        raise ValueError(f"Failed to parse line: '{line}'")
    winning_numbers = set((int(n)
                          for n in re.findall(r'\d+', parsed.group(2))))
    selected_numbers = (int(n)
                        for n in re.findall(r'\d+', parsed.group(3)))
    sum = 0
    for n in selected_numbers:
        if n in winning_numbers:
            if sum == 0:
                sum = 1
            else:
                sum *= 2
    return sum


def part_one(lines: List[str]) -> int:
    values = [parse_card_value(line) for line in lines]
    return sum(values)


def part_two(lines: List[str]) -> int:
    values = [int(LINE_REGEX.match(line).group(1)) for line in lines]
    return sum(values)


def test_part_one():
    lines = open(path.Path(__file__).abspath().parent /
                 ('example.txt'), 'r').readlines()
    assert part_one(lines) == 13


if __name__ == '__main__':
    lines = open(path.Path(__file__).abspath().parent /
                 ('input.txt'), 'r').readlines()
    print('1.', part_one(lines))
    print('2.', part_two(lines))
