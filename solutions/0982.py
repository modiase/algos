"""
A strobogrammatic number is a positive number that appears the same after being
rotated 180 degrees. For example, 16891 is strobogrammatic. Create a program that
finds all strobogrammatic numbers with N digits.

Solved: ~5m
"""
import itertools


def find_all(n):
    digits = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
    all_permutations = itertools.product(digits, repeat=n)
    filtered_permutations = [
        p for p in all_permutations if ''.join(p).lstrip('0') == ''.join(p)]
    found = []
    for p in filtered_permutations:
        s = ''.join(p)
        if s == ''.join(reversed(s)):
            found.append(s)
    return found[:-1]


print(find_all(4))
print(find_all(3))
