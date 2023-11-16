"""
Given a string of parentheses, find the balanced string that can be produced from it using the minimum number of insertions and deletions. If there are multiple solutions, return any of them.

For example, given "(()", you could return "(())". Given "))()(", you could return "()()()()".


Completed: ~70m
"""


from itertools import count, combinations


def is_balanced(s):
    if len(s) == 0:
        return True
    o = [1 if c == '(' else -1 for c in s]
    if not (sum(o) == 0 and o[0] != -1 and o[-1] != 1):
        return False
    for i in range(1, len(s)):
        if sum(o[:i]) == 0:
            return is_balanced(s[i:])
    return True


def try_insertions(s):
    if is_balanced(s):
        return s

    def _try_insertions(s1, t):
        if is_balanced(s1):
            return s1
        if not t:
            return None
        t = list(sorted(t))
        for i in t:
            attempt = s1[:i] + ')' + s1[i:]
            reduced = [v for v in t if v != i]
            r = _try_insertions(attempt, reduced)
            if r is not None:
                return r

            attempt = s1[:i] + '(' + s1[i:]
            reduced = [v for v in t if v != i]
            r = _try_insertions(attempt, reduced)
            if r is not None:
                return r
        return None

    result = None
    for i in count(1):
        comb = list(combinations(range(len(s) + i), i))
        for c in comb:
            attempt = _try_insertions(s, c)
            if attempt is not None:
                result = attempt
                break
        if result is not None:
            break
    return result


print(try_insertions(')))()((()('))
