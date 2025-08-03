import math
import json
import sys
from pathlib import Path
from typing import List


class NeverException(RuntimeError):
    def __init__(self):
        super().__init__("This should never occur")


def find_median_sorted_arrays(nums1: List[int], nums2: List[int]) -> float:
    if len(nums1) < len(nums2):
        return find_median_sorted_arrays(nums2, nums1)

    m = len(nums1)
    n = len(nums2)
    s = m + n
    c = (s // 2) + 1
    step = (n // 2) + 1
    nn = n
    nm = c - nn

    M = nums1
    N = nums2

    if m == 0:
        return 0.0
    if n == 0:
        if s % 2 == 0:
            return (M[nm - 1] + M[nm - 2]) / 2
        else:
            return M[nm - 1]
    elif m == 1:
        return (M[0] + N[0]) / 2
    else:
        while 1:
            pm = nm - 1
            pn = nn - 1

            a = True
            if pn != 0:
                a = N[pn - 1] <= M[pm]
            b = pm == 0 or (M[pm - 1] <= N[pn])

            if a and b:
                lower_median = min(M[pm], N[pn])
                if lower_median == M[pm]:
                    upper_median = N[pn]
                    if pm < m - 1:
                        upper_median = min(N[pn], M[pm + 1])
                else:
                    upper_median = M[pm]
                    if pn < n - 1:
                        upper_median = min(M[pm], N[pn + 1])

                if s % 2 == 0:
                    return (lower_median + upper_median) / 2
                else:
                    return upper_median

            else:
                if not a:
                    if nn == 1:  # => pn == 0
                        l = N[0]
                        if pm != 0:
                            l = max(N[0], M[pm - 1])

                        lower_median = min(l, M[pm])
                        if lower_median == M[pm]:
                            upper_median = N[0]
                            if pm < m - 1:
                                upper_median = min(M[pm + 1], N[0])
                        else:
                            upper_median = M[pm]

                        if s % 2 == 0:
                            return (lower_median + upper_median) / 2
                        else:
                            return upper_median
                    nn = max(nn - step, 1)
                elif not b:
                    if nn == n:
                        l = max(N[pn], M[pm - 1])

                        lower_median = min(l, M[pm])
                        if lower_median == M[pm]:
                            upper_median = min(M[pm + 1], N[n - 1])
                        else:
                            upper_median = M[pm]

                        if s % 2 == 0:
                            return (lower_median + upper_median) / 2
                        else:
                            return upper_median
                    nn = min(nn + step, n)
                else:
                    raise NeverException
                nm = c - nn
                step = math.ceil(step / 2)

        raise NeverException


def make_test_case(arr1: List[int], arr2: List[int]):
    l = list(sorted(arr1 + arr2))
    N = len(l)
    m = N // 2
    if N == 0:
        expected = 0.0
    elif N % 2 == 0:
        expected = (l[m] + l[m - 1]) / 2
    else:
        expected = float(l[m])

    result = find_median_sorted_arrays(arr1, arr2)
    message = f"arr1: {arr1}, arr2: {arr2}, expected: {expected}, result: {result}"
    assert result == expected, f"FAILED: {message}"


def test_cases():
    data = [
        ([], []),
        ([1], []),
        ([1], [2]),
        ([1], [2, 3]),
        ([1], [2, 3, 4]),
        ([1, 2], [2, 3]),
        ([1, 2], [3, 4, 5, 6]),
        ([1, 2, 3], [1, 2, 3]),
        ([1, 2], [1, 2, 3]),
        ([1, 2, 3], [4, 5, 6]),
        ([-1, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0]),
        ([-1, 1], [0, 0, 0, 0]),
    ]
    for d in data:
        make_test_case(*d)


def s_to_l(s):
    return [int(x) for x in s.split(" ")]


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("<script> <nums1_filepath> <nums2_filepath>")
        sys.exit(1)
    else:
        m = (
            Path(sys.argv[1]).is_file() and s_to_l(Path(sys.argv[1]).read_text())
        ) or s_to_l(sys.argv[1])
        n = (
            Path(sys.argv[2]).is_file() and s_to_l(Path(sys.argv[2]).read_text())
        ) or s_to_l(sys.argv[2])
        print(find_median_sorted_arrays(m, n))
