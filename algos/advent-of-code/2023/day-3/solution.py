import re
import string
import path
from typing import Dict, List, Tuple, TypeVar


def parse_line(line: str) -> List[Tuple[int, str]]:
    result = []
    s = ""
    h = 0
    count = 1
    for c in line.strip():
        if c != "." and c not in string.digits:
            if s:
                result.append((h, s))
                s = ""
                h = count
            result.append((count, c))
            count += 1
            continue
        else:
            if s:
                if c == "." or c not in string.digits:
                    result.append((h, s))
                    s = ""
                else:
                    s += c
            else:
                if c == ".":
                    count += 1
                    continue
                s = c
                h = count
            count += 1

    if s:
        result.append((h, s))
    return result


T = TypeVar("T")


def create_sparse_matrix(entries: List[Tuple[int, int, T]]) -> Dict[int, Dict[int, T]]:
    result = {}
    for row, col, entry in entries:
        rowdict = result.setdefault(row, {})
        rowdict[col] = entry
    return result


def get_symbols(
    processed_lines: List[Tuple[int, List[Tuple[int, str]]]],
) -> List[Tuple[int, int, str]]:
    result = []
    for row, ss in processed_lines:
        for col, s in ss:
            if not all([c in string.digits for c in s]):
                result.append((row, col, s))
    return result


def get_numbers(
    processed_lines: List[Tuple[int, List[Tuple[int, str]]]],
) -> List[Tuple[int, int, str]]:
    result = []
    for row, ss in processed_lines:
        for col, s in ss:
            if all([c in string.digits for c in s]):
                result.append((row, col, s))
    return result


def get_spread_numbers(
    processed_lines: List[Tuple[int, List[Tuple[int, str]]]],
) -> List[Tuple[int, int, Tuple[int, str]]]:
    result = []
    for row, ss in processed_lines:
        for col, s in ss:
            if all([c in string.digits for c in s]):
                for j in range(0, len(s)):
                    result.append((row, col + j, (col, s)))

    return result


def get_neighbours(row: int, col: int, number: str):
    def _get_neighbours(row: int, col: int) -> List[Tuple[int, int]]:
        neighbours = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                neighbours.append((i + row, j + col))
        return neighbours

    neighbours = []
    for digit, _ in enumerate(number, 0):
        neighbours += _get_neighbours(row, col + digit)
    return neighbours


def part_one(lines: List[str]):
    sum = 0
    processed_lines = [
        (lineno, parse_line(line)) for lineno, line in enumerate(lines, 1)
    ]
    symbols: List[Tuple[int, int, str]] = get_symbols(processed_lines)
    numbers: List[Tuple[int, int, str]] = get_numbers(processed_lines)

    symbol_matrix = create_sparse_matrix(symbols)
    for row, col, number in numbers:
        neighbours = get_neighbours(row, col, number)
        for nr, nc in neighbours:
            if (r := symbol_matrix.get(nr)) is not None:
                if r.get(nc) is not None:
                    sum += int(number)
                    break
    return sum


def part_two(lines: List[str]):
    sum = 0
    processed_lines = [
        (lineno, parse_line(line)) for lineno, line in enumerate(lines, 1)
    ]
    symbols: List[Tuple[int, int, str]] = get_symbols(processed_lines)
    numbers: List[Tuple[int, int, Tuple[int, str]]] = get_spread_numbers(
        processed_lines
    )

    numbers_matrix = create_sparse_matrix(numbers)
    for row, col, symbol in symbols:
        if symbol != "*":
            continue
        neighbours = get_neighbours(row, col, symbol)
        neighbouring_numbers = []
        for nr, nc in neighbours:
            if (
                (r := numbers_matrix.get(nr)) is not None
                and (number := r.get(nc)) is not None
                and number not in [x[2] for x in neighbouring_numbers]
            ):
                neighbouring_numbers.append((nr, nc, number))
        if len(neighbouring_numbers) == 2:
            a = int(neighbouring_numbers[0][2][1])
            b = int(neighbouring_numbers[1][2][1])
            sum += a * b
    return sum


def test_parse_line():
    lines = open(path.Path(__file__).abspath().parent / "example.txt", "r").readlines()
    for lineno, line in enumerate(lines, 1):
        processed_line = (lineno, parse_line(line))
        assert (lineno, [int(t[2]) for t in get_numbers([processed_line])]) == (
            lineno,
            [int(s) for s in re.findall(r"\d+", line.strip() + ".")],
        )


def test_parse_line_two():
    lines = open(
        path.Path(__file__).abspath().parent / "example-2.txt", "r"
    ).readlines()
    for lineno, line in enumerate(lines, 1):
        processed_line = (lineno, parse_line(line))
        try:
            assert [int(t[2]) for t in get_numbers([processed_line])] == [
                int(s) for s in re.findall(r"\d+", line)
            ]
        except AssertionError:
            print(lineno, line)
            raise


def test_example_part_one():
    lines = open(path.Path(__file__).abspath().parent / "example.txt", "r").readlines()
    assert part_one(lines) == 4361


def test_example_part_two():
    lines = open(path.Path(__file__).abspath().parent / "example.txt", "r").readlines()
    assert part_two(lines) == 467835


def test_example_2_part_one():
    lines = open(
        path.Path(__file__).abspath().parent / "example-2.txt", "r"
    ).readlines()
    assert part_one(lines) == 522726


if __name__ == "__main__":
    lines = open(path.Path(__file__).abspath().parent / "input.txt", "r").readlines()
    print(part_one(lines))
    print(part_two(lines))
