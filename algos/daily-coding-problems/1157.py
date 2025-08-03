"""
In academia, the h-index is a metric used to calculate the impact of a researcher
's papers. It is calculated as follows:

A researcher has index h if at least h of her N papers have h citations each. If
there are multiple h satisfying this formula, the maximum is chosen.

For example, suppose N = 5, and the respective citations of each paper are [4, 3,
0, 1, 5]. Then the h-index would be 3, since the researcher has 3 papers with at
least 3 citations.

Given a list of paper citations of a researcher, calculate their h-index.
"""

from typing import Dict, List, Tuple
from itertools import count


def compute_h_index(citations: List[int]) -> int:
    counts: Dict[int, int] = dict()
    for citation in citations:
        if counts.get(citation) is None:
            counts[citation] = 1
        else:
            counts[citation] = counts[citation] + 1

    sorted_counts: List[Tuple[int, int]] = list(sorted(counts.items(), reverse=True))
    cumulative_counts = [sorted_counts[0]]

    for citations, ct in sorted_counts[1:]:
        cumulative_counts.append((citations, ct + cumulative_counts[-1][1]))

    l = []
    N = len(cumulative_counts)
    for idx, value in enumerate(cumulative_counts, 0):
        citations, count = value
        l.append((citations, count))
        if idx < N - 1:
            a = 1
            while citations - a > cumulative_counts[idx + 1][0]:
                l.append((citations - a, count))
                a += 1
    smallest = cumulative_counts[-1]
    a = 1
    while (current := smallest[0] - a) > 0:
        l.append((current, smallest[1]))
        a += 1

    for citations, count in l:
        if citations <= count:
            return citations

    raise Exception("absurd")


assert compute_h_index([4, 3, 0, 1, 5]) == 3
assert compute_h_index([5]) == 1
assert compute_h_index([4, 4, 4]) == 3
assert compute_h_index([4, 4, 4, 4]) == 4
assert compute_h_index([4, 4, 4, 4, 4]) == 4
assert compute_h_index([1, 1, 1, 1, 1, 5]) == 1
assert compute_h_index([1, 1, 1, 1, 5, 5]) == 2
assert compute_h_index([1, 1, 1, 5, 5, 5]) == 3
assert compute_h_index([1, 1, 5, 5, 5, 5]) == 4
assert compute_h_index([1, 5, 5, 5, 5, 5]) == 5


"""
Better solution
"""


def h_index(citations):
    n = len(citations)
    citations.sort(reverse=True)

    h = 0
    while h < n and citations[h] >= h + 1:
        h += 1

    return h
