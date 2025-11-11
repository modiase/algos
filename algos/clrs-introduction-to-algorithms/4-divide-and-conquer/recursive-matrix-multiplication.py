#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.pytest -p python313Packages.click -p python313Packages.loguru
from __future__ import annotations

import array
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

import click
import pytest
from loguru import logger


@dataclass(frozen=True, slots=True, kw_only=True)
class Matrix:
    rows: int
    cols: int
    data: array.array

    def __post_init__(self):
        if len(self.data) != self.rows * self.cols:
            raise ValueError("Data size doesn't match dimensions")

    def __getitem__(self, pos: tuple[int, int]) -> float:
        row, col = pos
        return self.data[row * self.cols + col]

    @classmethod
    def of(cls, data: Sequence[Sequence[float | int]]) -> Self:
        if not data:
            raise ValueError("Empty sequence")
        return cls(
            rows=len(data),
            cols=len(data[0]),
            data=array.array("d", [item for row in data for item in row]),
        )

    def to_lists(self) -> Sequence[Sequence[float]]:
        return [[self[i, j] for j in range(self.cols)] for i in range(self.rows)]

    def view(
        self, row_start: int, row_end: int, col_start: int, col_end: int
    ) -> Matrix.View:
        return Matrix.View(
            parent_data=self.data,
            parent_cols=self.cols,
            row_offset=row_start,
            col_offset=col_start,
            rows=row_end - row_start,
            cols=col_end - col_start,
        )

    @dataclass(frozen=True, slots=True, kw_only=True)
    class View:
        parent_data: array.array
        parent_cols: int
        row_offset: int
        col_offset: int
        rows: int
        cols: int

        def __getitem__(self, pos: tuple[int, int]) -> float:
            row, col = pos
            actual_row = self.row_offset + row
            actual_col = self.col_offset + col
            return self.parent_data[actual_row * self.parent_cols + actual_col]

        def view(
            self, row_start: int, row_end: int, col_start: int, col_end: int
        ) -> Matrix.View:
            return Matrix.View(
                parent_data=self.parent_data,
                parent_cols=self.parent_cols,
                row_offset=self.row_offset + row_start,
                col_offset=self.col_offset + col_start,
                rows=row_end - row_start,
                cols=col_end - col_start,
            )

        def to_lists(self) -> Sequence[Sequence[float]]:
            return [[self[i, j] for j in range(self.cols)] for i in range(self.rows)]


type MatrixLike = Matrix | Matrix.View


def matrix_combine(
    c11: MatrixLike, c12: MatrixLike, c21: MatrixLike, c22: MatrixLike
) -> Matrix:
    total_rows = c11.rows + c21.rows
    total_cols = c11.cols + c12.cols
    logger.trace(
        f"Combining quadrants: {c11.rows}x{c11.cols}, {c12.rows}x{c12.cols}, "
        f"{c21.rows}x{c21.cols}, {c22.rows}x{c22.cols} into {total_rows}x{total_cols}"
    )
    data = array.array("d", [0.0] * (total_rows * total_cols))

    for i in range(c11.rows):
        for j in range(c11.cols):
            data[i * total_cols + j] = c11[i, j]

    for i in range(c12.rows):
        for j in range(c12.cols):
            data[i * total_cols + (j + c11.cols)] = c12[i, j]

    for i in range(c21.rows):
        for j in range(c21.cols):
            data[(i + c11.rows) * total_cols + j] = c21[i, j]

    for i in range(c22.rows):
        for j in range(c22.cols):
            data[(i + c11.rows) * total_cols + (j + c11.cols)] = c22[i, j]

    return Matrix(rows=total_rows, cols=total_cols, data=data)


def matrix_add(a: MatrixLike, b: MatrixLike) -> Matrix:
    if a.rows != b.rows or a.cols != b.cols:
        raise ValueError("Matrix dimensions must match")

    logger.trace(f"Adding two {a.rows}x{a.cols} matrices")
    data = array.array("d")
    for i in range(a.rows):
        for j in range(a.cols):
            data.append(a[i, j] + b[i, j])
    return Matrix(rows=a.rows, cols=a.cols, data=data)


def matrix_multiply_base(a: MatrixLike, b: MatrixLike) -> Matrix:
    if a.cols != b.rows:
        raise ValueError("Matrix dimensions incompatible for multiplication")

    logger.trace(f"Base case: multiplying {a.rows}x{a.cols} × {b.rows}x{b.cols}")
    data = array.array("d", [0.0] * (a.rows * b.cols))
    for i in range(a.rows):
        for j in range(b.cols):
            total = 0.0
            for k in range(a.cols):
                total += a[i, k] * b[k, j]
            data[i * b.cols + j] = total

    return Matrix(rows=a.rows, cols=b.cols, data=data)


def matrix_multiply_recursive(
    a: MatrixLike, b: MatrixLike, threshold: int = 2, depth: int = 0
) -> Matrix:
    indent = "  " * depth
    logger.trace(f"{indent}Recursive multiply: {a.rows}x{a.cols} × {b.rows}x{b.cols}")

    if a.cols != b.rows:
        raise ValueError("Matrix dimensions incompatible for multiplication")

    if a.rows <= threshold or a.cols <= threshold or b.cols <= threshold:
        logger.trace(f"{indent}→ Reached threshold, using base case")
        return matrix_multiply_base(a, b)

    mid_row_a = a.rows // 2
    mid_col_a = a.cols // 2
    mid_col_b = b.cols // 2

    logger.trace(f"{indent}Partitioning into quadrants")
    a11 = a.view(0, mid_row_a, 0, mid_col_a)
    a12 = a.view(0, mid_row_a, mid_col_a, a.cols)
    a21 = a.view(mid_row_a, a.rows, 0, mid_col_a)
    a22 = a.view(mid_row_a, a.rows, mid_col_a, a.cols)

    b11 = b.view(0, mid_col_a, 0, mid_col_b)
    b12 = b.view(0, mid_col_a, mid_col_b, b.cols)
    b21 = b.view(mid_col_a, b.rows, 0, mid_col_b)
    b22 = b.view(mid_col_a, b.rows, mid_col_b, b.cols)

    logger.trace(f"{indent}Computing C11 = A11*B11 + A12*B21")
    c11 = matrix_add(
        matrix_multiply_recursive(a11, b11, threshold, depth + 1),
        matrix_multiply_recursive(a12, b21, threshold, depth + 1),
    )
    logger.trace(f"{indent}Computing C12 = A11*B12 + A12*B22")
    c12 = matrix_add(
        matrix_multiply_recursive(a11, b12, threshold, depth + 1),
        matrix_multiply_recursive(a12, b22, threshold, depth + 1),
    )
    logger.trace(f"{indent}Computing C21 = A21*B11 + A22*B21")
    c21 = matrix_add(
        matrix_multiply_recursive(a21, b11, threshold, depth + 1),
        matrix_multiply_recursive(a22, b21, threshold, depth + 1),
    )
    logger.trace(f"{indent}Computing C22 = A21*B12 + A22*B22")
    c22 = matrix_add(
        matrix_multiply_recursive(a21, b12, threshold, depth + 1),
        matrix_multiply_recursive(a22, b22, threshold, depth + 1),
    )

    logger.trace(f"{indent}Combining quadrants")
    return matrix_combine(c11, c12, c21, c22)


@pytest.mark.parametrize(
    "a_data, b_data, expected",
    [
        (
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
            [[19, 22], [43, 50]],
        ),
        (
            [[1, 0], [0, 1]],
            [[5, 6], [7, 8]],
            [[5, 6], [7, 8]],
        ),
        (
            [[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2]],
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
            [[2, 4, 6, 8], [10, 12, 14, 16], [18, 20, 22, 24], [26, 28, 30, 32]],
        ),
        (
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
        ),
        (
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        ),
        (
            [[1, 2], [3, 4], [5, 6]],
            [[1, 2, 3], [4, 5, 6]],
            [[9, 12, 15], [19, 26, 33], [29, 40, 51]],
        ),
    ],
)
def test_multiply_recursive(
    a_data: Sequence[Sequence[int]],
    b_data: Sequence[Sequence[int]],
    expected: Sequence[Sequence[int]],
) -> None:
    a = Matrix.of(a_data)
    b = Matrix.of(b_data)
    result = matrix_multiply_recursive(a, b)
    assert result.to_lists() == expected


@pytest.mark.parametrize(
    "a_data, b_data, expected",
    [
        (
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
            [[19, 22], [43, 50]],
        ),
        (
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8], [9, 10], [11, 12]],
            [[58, 64], [139, 154]],
        ),
    ],
)
def test_multiply_iterative(
    a_data: Sequence[Sequence[int]],
    b_data: Sequence[Sequence[int]],
    expected: Sequence[Sequence[int]],
) -> None:
    a = Matrix.of(a_data)
    b = Matrix.of(b_data)
    result = matrix_multiply_base(a, b)
    assert result.to_lists() == expected


def test_matrix_view() -> None:
    m = Matrix.of([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])

    top_left = m.view(0, 2, 0, 2)
    assert top_left.to_lists() == [[1, 2], [5, 6]]

    top_right = m.view(0, 2, 2, 4)
    assert top_right.to_lists() == [[3, 4], [7, 8]]

    bottom_left = m.view(2, 4, 0, 2)
    assert bottom_left.to_lists() == [[9, 10], [13, 14]]

    bottom_right = m.view(2, 4, 2, 4)
    assert bottom_right.to_lists() == [[11, 12], [15, 16]]

    sub_view = top_left.view(0, 1, 0, 1)
    assert sub_view.to_lists() == [[1]]


cli = click.Group()


@cli.command()
@click.option("--size", "-n", default=4, help="Size of square matrices to multiply")
@click.option(
    "--threshold", "-t", default=2, help="Threshold for switching to iterative"
)
def demo(size: int, threshold: int) -> None:
    logger.remove()
    logger.add(lambda msg: click.echo(msg, err=True), level="TRACE", colorize=True)
    logger.info("=" * 60)
    logger.info("RECURSIVE MATRIX MULTIPLICATION")
    logger.info("=" * 60)

    rng = random.Random(42)
    a_data = [[rng.randint(0, 9) for _ in range(size)] for _ in range(size)]
    b_data = [[rng.randint(0, 9) for _ in range(size)] for _ in range(size)]

    a = Matrix.of(a_data)
    b = Matrix.of(b_data)

    click.echo(f"\nMatrix A ({size}x{size}):")
    for row in a.to_lists():
        click.echo(f"  {row}")

    click.echo(f"\nMatrix B ({size}x{size}):")
    for row in b.to_lists():
        click.echo(f"  {row}")

    logger.info(f"\nStarting recursive multiplication (threshold={threshold})...")

    result = matrix_multiply_recursive(a, b, threshold)

    logger.info("\n" + "=" * 60)
    logger.success("Multiplication complete!")
    logger.info("=" * 60)

    click.echo(f"\nResult ({size}x{size}):")
    for row in result.to_lists():
        click.echo(f"  {row}")


@cli.command("test")
def run_tests() -> None:
    logger.disable("")
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    cli()
