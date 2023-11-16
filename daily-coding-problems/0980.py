"""
On our special chessboard, two bishops attack each other if they share the same diagonal. This includes bishops that have another bishop located between them, i.e. bishops can attack through pieces.

You are given N bishops, represented as (row, column) tuples on a M by M chessboard. Write a function to count the number of pairs of bishops that attack each other. The ordering of the pair doesn't matter: (1, 2) is considered the same as (2, 1).

For example, given M = 5 and the list of bishops:

(0, 0)
(1, 2)
(2, 2)
(4, 0)
The board would look like this:

[b 0 0 0 0]
[0 0 b 0 0]
[0 0 b 0 0]
[0 0 0 0 0]
[b 0 0 0 0]
You should return 2, since bishops 1 and 3 attack each other, as well as bishops 3 and 4.

Solved: 28m
"""


def attacks(M, bs):
    count = 0
    for i in range(len(bs)):
        b = bs[i]
        attacking = [(b[0] + i, b[1] + i) for i in range(1, min(M-b[1], M-b[0]))] + \
                    [(b[0] - i, b[1] - i) for i in range(1, min(b[1]+1, b[0]+1))] + \
                    [(b[0] + i, b[1] - i) for i in range(1, min(b[1]+1, M-b[0]))] + \
                    [(b[0] - i, b[1] + i)
                     for i in range(1, min(b[0]+1, M-b[1]))]  # no-wrap
        for ib in range(i, len(bs)):
            ob = bs[ib]
            if ob in attacking:
                count += 1
    return count


assert attacks(5, [(0, 0), (1, 2), (2, 2), (4, 0)]) == 2
