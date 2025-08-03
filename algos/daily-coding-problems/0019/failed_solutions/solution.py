from utils import Node, read_mat_from_file
from typing import List, Tuple

Matrix = List[List[float]]

logfile = open("out.log", "w+")


def lowest_choices(m: Matrix, idx: Tuple[int, int]) -> Matrix:
    """Returns two choices, one where the left column is moved, the other
    where the right column is moved."""
    i_elems = [m[idx[1]][x] for x in range(len(m[idx[1]]))]
    ipls_elems = [m[idx[1] + 1][x] for x in range(len(m[idx[1]]))]

    choice_one = i_elems.index(min(*i_elems))
    if choice_one == idx[0]:
        cpy = [*i_elems]
        del cpy[idx[0]]
        choice_one = i_elems.index(min(*cpy))

    choice_two = ipls_elems.index(min(*ipls_elems))
    if choice_two == idx[0]:
        cpy = [*ipls_elems]
        del cpy[idx[0]]
        choice_two = ipls_elems.index(min(*cpy))
    return (choice_one, choice_two)


def greedy_move(m: Matrix, i: Matrix, idx: Tuple[int, int]) -> Matrix:
    """Generates a greedy move and returns the corresponding solution matrix."""
    choices = lowest_choices(m, idx)
    diff1 = m[idx[1]][idx[0]] - m[idx[1]][choices[0]]
    diff2 = m[idx[1]][idx[0] + 1] - m[idx[1]][choices[0] + 1]
    if diff1 < diff2:
        i[idx[0]][idx[1]] = 0
        i[choices[0]][idx[1]] = 1.0
    else:
        i[idx[0]][idx[1] + 1] = 0
        i[choices[1]][idx[1] + 1] = 1.0
    return i


def non_greedy_move(m: Matrix, i: Matrix, idx: int) -> Matrix:
    """Generates a non greedy move and returns the corresponding solution matrix."""
    choices = lowest_choices(m, idx)
    diff1 = m[idx[1]][idx[0]] - m[idx[1]][choices[0]]
    diff2 = m[idx[1]][idx[0] + 1] - m[idx[1]][choices[0] + 1]
    if diff1 >= diff2:
        i[idx[0]][idx[1]] = 0
        i[choices[0]][idx[1]] = 1.0
    else:
        i[idx[0]][idx[1] + 1] = 0
        i[choices[1]][idx[1] + 1] = 1.0
    print(i, is_valid(i), file=logfile)
    return i


def is_valid(N: Matrix) -> bool:
    """True if and only if a solution matrix - i.e., one composed entirely
    of ones and zeros, with a one in each row - does not contain any adjacent columns
    which contain a one in the same row such that when matrix multiplied into the colour
    matrix two columns have the same colour"""
    for i in range(len(N) - 1):
        for j in range(len(N[0]) - 1):
            if N[i][j] and N[i][j + 1]:
                return (i, j)
    return True


def mat_tot(m1: Matrix, m2: Matrix) -> float:
    """multiply two matrices to find the sum of the elements of the vector
    produced.

    m: left matrix
    N: right matrix
    returns: sum of the elements of v where v is the vector produced.
    """
    assert len(m1) == len(m2[0])
    assert len(m1[0]) == len(m2)
    total = 0
    for i in range(len(m1)):
        for j in range(len(m1[0])):
            total += m1[i][j] * m2[j][i]
    return total


def initialise_solution(m: Matrix) -> Matrix:
    """Initialises the solution by selecting the lowest
    cost price in each column.

    m: The input matrix
    returns: A matrix which has the minimal value for mat_tot(n,m)
    """
    N = len(m)
    K = len(m[0])
    n = [l := [0] * N for _ in range(0, K)]
    for row in range(len(m)):
        n[m[row].index(min(*m[row]))][row] = 1.0
    return n


def generate_tree(
    m: Matrix,
    i: Matrix,
) -> Node:
    """Moves indexes to generate a tree of transformations which
    terminate at leaf nodes containing valid states. Each node is a
    tuple which has two elements:
        - the current proposed solution.
        - a boolean which is True if the solution is valid.

    returns: The root of the tree with the above described property.
    """
    root = Node((i, m))

    idx = is_valid(i)
    if idx is True:
        return root

    root.l = generate_tree(m, greedy_move(m, i, idx))
    root.r = generate_tree(m, non_greedy_move(m, i, idx))

    return root


def lowest_solution(m: Matrix, t: Node) -> Node:
    """Returns the lowest solution matrix in the tree t when that
    solution is combined using mat_tot with m."""
    if not t.l and not t.r:
        return t
    l = lowest_solution(m, t.l)
    r = lowest_solution(m, t.r)
    return l if mat_tot(*l.payload) < mat_tot(*r.payload) else r


def main(m: Matrix) -> Matrix:
    i = initialise_solution(m)

    t = generate_tree(m, i)

    i0 = lowest_solution(m, t).payload[0]

    return i0


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("in_fp", type=str, help="File containing the input matrix.")
    args = parser.parse_args()
    mat = read_mat_from_file(args.in_fp)
    res = main(mat)
