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
from collections.abc import MutableSequence


def url_encode_spaces(input: MutableSequence[str], n: int) -> None:
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


if __name__ == '__main__':
    test_cases = [
        (list('Mr John Smith    '), 13, 'Mr%20John%20Smith'),
        (list('   '), 1, '%20'),
        (list(''), 0, ''),
        (list('abc   '), 4, 'abc%20'),
        (list('abc'), 3, 'abc'),
        (list('abc def  '), 7, r'abc%20def'),
        (list('abc def ghi    '), 11, 'abc%20def%20ghi'),
        (list('abc def ghi jkl      '), 15, 'abc%20def%20ghi%20jkl'),
        (list('abc def ghi jkl mno        '), 19, 'abc%20def%20ghi%20jkl%20mno'),
        
    ]
    
    for test_case in test_cases:
        input, n, expected = test_case
        og_input = input[:]
        url_encode_spaces(input, n)
        assert (got:=''.join(input)) == expected, f"{og_input=}: {got=} != {expected=}"
