import random as rn


def swap(A, i, j):
    tmp = A[j]
    A[j] = A[i]
    A[i] = tmp


def partition(A, p, lb, rb):
    x = A[p]

    swap(A, p, rb)

    i = lb - 1
    for lp in range(lb, rb - 1):
        if A[lp] <= x:
            i += 1
            swap(A, i, lp)

    swap(A, i + 1, rb)
    return i + 1


def _select(A, lb, rb, i):
    N = len(A)
    o = i + 1
    if o < 1:
        raise ValueError(f"Order statistic must be positive. Got {o}")
    if N <= lb:
        raise ValueError(
            f"Cannot select {o}{'st' if o == 1 else 'nd' if o == 2 else 'th'} order statistic from array of length {N=}."
        )

    if rb == i:
        return A[rb]
    q = partition(A, lb, rb, rn.randint(lb, rb))
    k = q - lb + 1
    if lb == k:
        return A[q]
    elif lb < k:
        return _select(A, rb, q - 1, lb)
    return _select(A, q + 1, i, lb - k)


def select(A, i):
    return _select(A, 0, len(A) - 1, i)


if __name__ == "__main__":
    A = list(range(1, 11))
    rn.shuffle(A)
    N = len(A) - 1
    print(A)
    print(s := rn.randint(0, N))
    print(partition(A, s, 0, N))
    print(select(A, s))
