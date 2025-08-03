import logging
from typing import cast, Generator, List, Optional, Tuple


TTile = Tuple[int, int]
TMatrix = List[List[bool]]


def tile_is_not_wall(tile: TTile, matrix: TMatrix) -> bool:
    return not matrix[tile[0]][tile[1]]


def tile_is_within_matrix_limits(tile: TTile, matrix: TMatrix) -> bool:
    if tile[0] < 0 or tile[1] < 0:
        return False
    return tile[0] < len(matrix) and tile[1] < len(matrix[tile[0]])


def generate_moves(start: TTile) -> Generator[TTile, None, None]:
    x, y = start
    yield (x - 1, y)
    yield (x + 1, y)
    yield (x, y - 1)
    yield (x, y + 1)


def valid_moves(start: TTile, matrix: TMatrix) -> List[TTile]:
    return [
        move
        for move in generate_moves(start=start)
        if tile_is_within_matrix_limits(tile=move, matrix=matrix)
        and tile_is_not_wall(tile=move, matrix=matrix)
    ]


def compute_all_paths(
    start: TTile, end: TTile, matrix: TMatrix
) -> Optional[List[List[TTile]]]:
    def _compute_all_paths(
        start: TTile, end: TTile, matrix: TMatrix, history: List[TTile]
    ) -> Optional[List[List[TTile]]]:
        if start == end:
            return [[end]]

        next_starts = [
            next_start
            for next_start in valid_moves(start, matrix=matrix)
            if next_start not in history
        ]

        if next_starts == []:
            return None

        paths: List[List[TTile]] = []
        for next_start in next_starts:
            computed_paths = _compute_all_paths(
                start=next_start, end=end, matrix=matrix, history=[*history, next_start]
            )
            if not computed_paths:
                continue
            for path in computed_paths:
                paths.append(path)

        if not paths:
            return None

        return [[start, *path] for path in paths]

    return _compute_all_paths(start=start, end=end, matrix=matrix, history=[start])


def main(start: TTile, end: TTile, matrix: TMatrix) -> Optional[int]:
    paths = compute_all_paths(start=start, end=end, matrix=matrix)
    if paths == []:
        return None
    paths = cast(List[List[TTile]], paths)
    length_of_shortest_path = min([len(path) for path in paths])
    return length_of_shortest_path - 1


logging.basicConfig(
    format="%(asctime)s - %(lineno)d - %(funcName)s: %(message)s", level=logging.DEBUG
)
