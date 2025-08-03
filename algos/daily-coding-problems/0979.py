"""
Conway's Game of Life takes place on an infinite two-dimensional board of square cells. Each cell is either dead or alive, and at each tick, the following rules apply:

Any live cell with less than two live neighbours dies.
Any live cell with two or three live neighbours remains living.
Any live cell with more than three live neighbours dies.
Any dead cell with exactly three live neighbours becomes a live cell.
A cell neighbours another cell if it is horizontally, vertically, or diagonally adjacent.

Implement Conway's Game of Life. It should be able to be initialized with a starting list of live cell coordinates and the number of steps it should run for. Once initialized, it should print out the board state at each step. Since it's an infinite board, print out only the relevant coordinates, i.e. from the top-leftmost live cell to bottom-rightmost live cell.

You can represent a live cell with an asterisk (*) and a dead cell with a dot (.).

Solved: 32m (including ~ 10m playing with the solution)
"""

import time


def get_neighbours(cell, board):
    m = len(board)
    neighbours = [
        (cell[0] + i, cell[1] + j) for i in range(-1, 2) for j in range(-1, 2)
    ]
    neighbours = [n for n in neighbours if n != cell]
    for i, neighbour in enumerate(neighbours):
        if neighbour[0] < 0:
            neighbours[i] = (neighbours[i][0] + m, neighbours[i][1])
        if neighbour[0] >= m:
            neighbours[i] = (neighbours[i][0] - m, neighbours[i][1])
        if neighbour[1] < 0:
            neighbours[i] = (neighbours[i][0], neighbours[i][1] + m)
        if neighbour[1] >= m:
            neighbours[i] = (neighbours[i][0], neighbours[i][1] - m)
    return neighbours


def get_live_neighbours(cell, board):
    count = 0
    for neighbour in get_neighbours(cell, board):
        if board[neighbour[0]][neighbour[1]]:
            count += 1
    return count


def is_alive(cell, board):
    return board[cell[0]][cell[1]]


def is_survivor(cell, board):
    neighbours = get_live_neighbours(cell, board)
    return (is_alive(cell, board) and (neighbours == 2 or neighbours == 3)) or (
        not is_alive(cell, board) and neighbours == 3
    )


def init_board(m):
    return [[False] * m for _ in range(m)]


def evolve_board(bn0):
    bn1 = init_board(len(bn0))
    for i in range(len(bn0)):
        for j in range(len(bn0)):
            if is_survivor((i, j), bn0):
                bn1[i][j] = True
    return bn1


def print_board(board):
    print(chr(27) + "[2J")
    for i in range(len(board)):
        print("| ", end="")
        row = []
        for j in range(len(board)):
            if board[i][j]:
                row.append("*")
            else:
                row.append(".")
        print(" ".join(row), end="")
        print(
            " |",
        )


M = 20
b0 = init_board(M)
b0[0][0] = True
b0[1][0] = True
b0[2][0] = True
b0[0][1] = True
b0[1][2] = True
while True:
    b0 = evolve_board(b0)
    print_board(b0)
    time.sleep(0.5)
