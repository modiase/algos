import random as rn

def swap(A, i, j):
    tmp = A[i]
    A[i] = A[j]
    A[j] = tmp

def partition(A, i, l, r):
    swap(A, i, r)

    lp, rp = l, r-1
    while lp <= rp:
        if A[lp] > A[r]:
            swap(A, lp, rp)
            rp -= 1
        else:
            lp += 1
    swap(A, r, rp+1)
    return rp + 1

def test_partition():
    N = 10
    A = [ rn.randint(-10,10) for _ in range(N) ]
    for i in range(N):
        A_cpy = A[:]
        q = partition(A_cpy, i, 0, N-1)
        assert all(v > A[i] for v in A_cpy[q+1:])
        assert all(v <= A[i] for v in A_cpy[:q])
