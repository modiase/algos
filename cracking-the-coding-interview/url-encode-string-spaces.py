"""
How can a string be URL encoded such that each
space is replaced by '%20'? The string is right
padded with spaces to the necessary length and 
the change should be done in-place. The 'true
length' (without padding) of the string is given.

e.g., input: 'Mr John Smith    ', 13 
gives 'Mr%20John%20Smith'

# Notes

## Summary
T: 25
C: Y
PD: 1

## Comments

Encountered an off-by-one error when not considering
termination condition. Also realised that when b caught
a we could stop and this avoided the problem at the front
of the array.

These sorts of algorithms seem useful for efficiently updating
a string since it is easiest to add space to arrays at the end.

tags: strings, in-place
"""
from typing import List


def url_encode_spaces(input: List[str], n: int) -> None:
    N = len(input)
    a = n - 1
    b = N - 1

    while b > a:
        if input[a] == ' ':
            b -= 2
        else:
            input[b] = input[a]
            input[a] = ' '
        b -= 1
        a -= 1

    a = 0
    while a < N:
        if input[a] == ' ':
            input[a] = '%'
            input[a+1] = '2'
            input[a+2] = '0'
            a += 2
        a += 1


def test_give_case():
    input, n = list('Mr John Smith    '), 13
    expected = 'Mr%20John%20Smith'
    url_encode_spaces(input, n)
    assert ''.join(input) == expected


def test_edge_case_one():
    input, n = list('   '), 1
    expected = '%20'
    url_encode_spaces(input, n)
    assert ''.join(input) == expected


def test_edge_case_two():
    input, n = list('abc   '), 4
    expected = 'abc%20'
    url_encode_spaces(input, n)
    assert ''.join(input) == expected
