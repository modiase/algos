"""
Suppose you are given two lists of n points, one list p1, p2, ..., pn on the line
y = 0 and the other list q1, q2, ..., qn on the line y = 1. Imagine a set of n line 
segments connecting each point pi to qi. Write an algorithm to determine how many 
pairs of the line segments intersect.
"""

def grad(a, b):
    (b[1] - a[1]) / (b[0] - a[0])

def intersect(p1, p2, q1, q2):
    mp = grad(p1, p2)
    mq = grad(q1, q2)

def solution(p, q) -> int:
    pass



if __name__ == '__main__':
    pass
