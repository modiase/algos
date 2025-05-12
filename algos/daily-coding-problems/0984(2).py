"""
Given a string of parentheses, find the balanced string that can be produced from it using the minimum number of insertions and deletions. If there are multiple solutions, return any of them.

For example, given "(()", you could return "(())". Given "))()(", you could return "()()()()".

"""
from itertools import count, permutations


def is_balanced(s):
    xs = [1 if x == '(' else -1 for x in s]
    if xs[-1] == 1:
        return False
    for i in range(len(xs)-1):
        if (sum(xs[:i]) == 0) and xs[i] == -1:
            return False
    return True
