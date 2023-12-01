from typing import List
import path
import string
import re

from utils import wordtodig


def part_one(lines: List[str]) -> int:
    sum = 0
    for line in lines:
        first_seen = None
        last_seen = None
        for c in line:
            if c in string.digits:
                if first_seen is None:
                    first_seen = c
                last_seen = c
        if first_seen is None or last_seen is None:
            raise RuntimeError("No digits found")
        sum += int(first_seen + last_seen)

    return (sum)


def coerce(s: str) -> str:
    if s in string.digits:
        return s
    return wordtodig[s]


def part_two(lines: List[str]) -> int:
    sum = 0
    sm = re.compile(f'(?=([0-9]|{"|".join(wordtodig.keys())}))')
    for line in lines:
        matches = sm.findall(line)
        sum += int(coerce(matches[0])+coerce(matches[-1]))
    return sum


def test_part_two():
    lines = open(path.Path(__file__).abspath().parent /
                 ('day-1-example.txt'), 'r').readlines()
    assert part_two(lines) == 281


if __name__ == '__main__':
    lines = open(path.Path(__file__).abspath().parent /
                 ('day-1.txt'), 'r').readlines()
    print('1.', part_one(lines))
    print('2.', part_two(lines))
