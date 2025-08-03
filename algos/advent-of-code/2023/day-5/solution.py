"""
--- Day 5: If You Give A Seed A Fertilizer ---
You take the boat and find the gardener right where you were told he would be: managing
a giant "garden" that looks more to you like a farm.

"A water source? Island Island is the water source!" You point out that Snow Island
isn't receiving any water.

"Oh, we had to stop the water because we ran out of sand to filter it with! Can't
make snow with dirty water. Don't worry, I'm sure we'll get more sand soon; we only
turned off the water a few days... weeks... oh no." His face sinks into a look of
horrified realization.

"I've been so busy making sure everyone here has food that I completely forgot to
check why we stopped getting more sand! There's a ferry leaving soon that is headed
over in that direction - it's much faster than your boat. Could you please go check it out?"

You barely have time to agree to this request when he brings up another. "While you
wait for the ferry, maybe you can help us with our food production problem. The latest
Island Island Almanac just arrived and we're having trouble making sense of it."

The almanac (your puzzle input) lists all of the seeds that need to be planted. It
also lists what type of soil to use with each kind of seed, what type of fertilizer
to use with each kind of soil, what type of water to use with each kind of fertilizer,
and so on. Every type of seed, soil, fertilizer and so on is identified with a
number, but numbers are reused by each category - that is, soil 123 and fertilizer
123 aren't necessarily related to each other.

For example:

seeds: 79 14 55 13

seed-to-soil map:
50 98 2
52 50 48

soil-to-fertilizer map:
0 15 37
37 52 2
39 0 15

fertilizer-to-water map:
49 53 8
0 11 42
42 0 7
57 7 4

water-to-light map:
88 18 7
18 25 70

light-to-temperature map:
45 77 23
81 45 19
68 64 13

temperature-to-humidity map:
0 69 1
1 0 69

humidity-to-location map:
60 56 37
56 93 4

The almanac starts by listing which seeds need to be planted: seeds 79, 14, 55, and 13.

The rest of the almanac contains a list of maps which describe how to convert numbers
from a source category into numbers in a destination category. That is, the section
that starts with seed-to-soil map: describes how to convert a seed number (the source)
to a soil number (the destination). This lets the gardener and his team know which
soil to use with which seeds, which water to use with which fertilizer, and so on.

Rather than list every source number and its corresponding destination number one
by one, the maps describe entire ranges of numbers that can be converted. Each line
within a map contains three numbers: the destination range start, the source range
start, and the range length.

Consider again the example seed-to-soil map:

50 98 2
52 50 48

The first line has a destination range start of 50, a source range start of 98, and
a range length of 2. This line means that the source range starts at 98 and contains
two values: 98 and 99. The destination range is the same length, but it starts at
50, so its two values are 50 and 51. With this information, you know that seed number
98 corresponds to soil number 50 and that seed number 99 corresponds to soil number 51.

The second line means that the source range starts at 50 and contains 48 values:
50, 51, ..., 96, 97. This corresponds to a destination range starting at 52 and
also containing 48 values: 52, 53, ..., 98, 99. So, seed number 53 corresponds to
soil number 55.

Any source numbers that aren't mapped correspond to the same destination number.
So, seed number 10 corresponds to soil number 10.

So, the entire list of seed numbers and their corresponding soil numbers looks like this:

seed  soil
0     0
1     1
...   ...
48    48
49    49
50    52
51    53
...   ...
96    98
97    99
98    50
99    51

With this map, you can look up the soil number required for each initial seed number:

Seed number 79 corresponds to soil number 81.
Seed number 14 corresponds to soil number 14.
Seed number 55 corresponds to soil number 57.
Seed number 13 corresponds to soil number 13.

The gardener and his team want to get started as soon as possible, so they'd like
to know the closest location that needs a seed. Using these maps, find the lowest
location number that corresponds to any of the initial seeds. To do this, you'll
need to convert each seed number through other categories until you can find its
corresponding location number. In this example, the corresponding types are:

Seed 79, soil 81, fertilizer 81, water 81, light 74, temperature 78, humidity 78, location 82.
Seed 14, soil 14, fertilizer 53, water 49, light 42, temperature 42, humidity 43, location 43.
Seed 55, soil 57, fertilizer 57, water 53, light 46, temperature 82, humidity 82, location 86.
Seed 13, soil 13, fertilizer 52, water 41, light 34, temperature 34, humidity 35, location 35.

So, the lowest location number in this example is 35.

What is the lowest location number that corresponds to any of the initial seed numbers?
"""

import re
import sys
from pathlib import Path
from typing import List, NamedTuple, Tuple


class RangeMap(NamedTuple):
    dest_start: int
    src_start: int
    size: int

    @property
    def dest_end(self):
        return self.dest_start + self.size

    @property
    def src_end(self):
        return self.src_start + self.size

    @property
    def src_range(self):
        return range(self.src_start, self.src_end)

    @property
    def dest_range(self):
        return range(self.dest_start, self.dest_end)


class Mapper:
    def __init__(self, ranges: List[RangeMap]):
        self._ranges: List[RangeMap] = []
        for rng_map in ranges:
            self._add_range(rng_map)

    def _add_range(self, rng_map: RangeMap):
        self._ranges.append(rng_map)
        self._normalize()

    def _normalize(self):
        self._ranges = sorted(self._ranges, key=lambda t: t[1])

    def map(self, src: int) -> int:
        for _dst, _src, _rng in self._ranges:
            if src >= _src and src < _src + _rng:
                return _dst + (src - _src)
        return src

    def map_range(self, src_rng: range) -> List[range]:
        mapped_ranges: List[range] = []

        s = src_rng.start
        for rng in self._ranges:
            # leave s -> rng.src_start unmapped since it is between RangeMaps
            if rng.src_start >= src_rng.stop:
                mapped_ranges.append(range(s, src_rng.stop))
                break
            mapped_ranges.append(range(s, rng.src_start))

            # map rng.src_start -> min(rng.src_end, src_rng.stop)
            o = 0 if rng.src_start >= src_rng.start else src_rng.start - rng.src_start
            s = min(rng.src_end, src_rng.stop)
            w = s - rng.src_start
            mapped_ranges.append(range(rng.dest_start + o, rng.dest_start + w))

        else:
            # map from the last stopping point to the end of the src_rng
            s = max(s, src_rng.start)
            mapped_ranges.append(range(s, src_rng.stop))

        mapped_ranges = sorted(
            [r for r in mapped_ranges if not r.start >= r.stop], key=lambda r: r.start
        )

        return mapped_ranges


class MapperChainer:
    _mappers: List[Mapper] = []

    def __init__(self, maps: List[Mapper]):
        self._mappers = maps

    def get(self, src: int) -> int:
        dst = src
        for mapper in self._mappers:
            dst = mapper.map(dst)
        return dst

    def chain(self, src_rng: range) -> List[range]:
        dsts = [src_rng]
        for mapper in self._mappers:
            tmp: List[range] = []
            for rng in dsts:
                tmp += mapper.map_range(rng)
            dsts = tmp
        return dsts


def parse_seeds_one(line: str) -> List[int]:
    seeds = [int(x) for x in re.match(r"seeds: (.*)", line).group(1).split(" ")]
    return seeds


def parse_maps(lines: List[str]) -> MapperChainer:
    N = len(lines)
    maps = []

    i = 0
    stack = []
    while not re.match(r".*map", lines[i]):
        i += 1
    while i < N:
        i += 1
        while lines[i] != "" and i < N:
            stack.append(lines[i])
            i += 1

        rngs: List[RangeMap] = []
        while stack:
            mapping = stack.pop()
            dst, src, rng = re.match(r"(\d+) (\d+) (\d+)", mapping).groups()
            rngs.append(RangeMap(int(dst), int(src), int(rng)))
        maps.append(Mapper(rngs))

        i += 1

    return MapperChainer(maps)


def parse_lines_one(lines: List[str]) -> Tuple[List[int], MapperChainer]:
    return parse_seeds_one(lines[0]), parse_maps(lines)


def part_one(lines: List[str]) -> int:
    seeds, chainer = parse_lines_one(lines)
    min_loc = sys.maxsize
    for seed in seeds:
        dest = chainer.get(seed)
        min_loc = min(dest, min_loc)
    return min_loc


def parse_seeds_two(line: str) -> List[range]:
    _seeds = parse_seeds_one(line)
    seeds: List[range] = []
    while _seeds:
        rng = _seeds.pop()
        start = _seeds.pop()
        seeds.append(range(start, start + rng))
    return seeds


def parse_lines_two(lines: List[str]) -> Tuple[List[range], MapperChainer]:
    return parse_seeds_two(lines[0]), parse_maps(lines)


def part_two(lines: List[str]) -> int:
    seeds, chainer = parse_lines_two(lines)
    min_loc = sys.maxsize
    for rng in seeds:
        min_loc = min(min_loc, min([r.start for r in chainer.chain(rng)]))
    return min_loc


if __name__ == "__main__":
    lines = Path("./input.txt").read_text().split("\n")
    # print(part_one(lines))
    print(part_two(lines))
