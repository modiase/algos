#!/usr/bin/env python3
from collections import Counter


def minWindow(s: str, t: str) -> str:
    """
    # naive
    # -----
    # let m = len(s), n = len(t) min window size n, max window size m 1 window
    # len m, 2 len m-1, k + 1, m - k = n => k = m - n and there are (m-n)+1
    # window len n total windows = sum_k=1^(m-n+1) = (m-n+1)(m-n+2)/2 = O(m^2)
    # + O(mn) + O(n^2) assume m > n => O(m^2)

    # better?
    # ------
    # construct an array containing the cumulative counter of each symbol in t
    # use the cumulative counters to construct O(1) range queries.
    # construct predecessor and successor arrays recursively shrink window
    # using predecessor and successor arrays to find next positions.
    """

    M, N = len(s), len(t)
    if M == 0 or N == 0:
        return ""
    symbols = set(t)
    cumulatives = [Counter({c: 0 for c in symbols})]

    def query(start: int, end: int) -> Counter:
        counter = cumulatives[end].copy()
        for c in symbols:
            counter[c] -= cumulatives[start][c]
        return counter

    for idx in range(M):
        counter = cumulatives[idx].copy()
        if (c := s[idx]) in symbols:
            counter[c] += 1

        cumulatives.append(counter)

    t_counter = Counter()
    for c in t:
        t_counter[c] += 1

    def satisfies(counter):
        has_excess = False
        for c in t_counter:  # Check all required characters first
            if counter[c] < t_counter[c]:
                return -1
            elif counter[c] > t_counter[c]:
                has_excess = True
        return 1 if has_excess else 0

    if satisfies(query(0, M)) < 0:
        return ""

    current = None
    predecessor = []
    for idx, c in enumerate(s):
        predecessor.append(current)
        if c in symbols:
            current = idx

    current = None
    successor: list[int | None] = []
    for idx, c in enumerate(reversed(s)):
        successor.append(current)
        if c in symbols:
            current = idx
    successor: list[int | None] = list(
        reversed([M - i - 1 if i is not None else None for i in successor])
    )

    _cache = {}

    def find_min_window(start: int | None, end: int | None) -> tuple[int, int] | None:
        if start is None or end is None:
            return None

        if (start, end) in _cache:
            return _cache[(start, end)]
        result = satisfies(query(start, end))

        if result < 0:  # Missing required characters, gone too far
            return None

        # result >= 0: Try to shrink the window
        best = (start, end)

        # Try sliding left boundary to the successor
        if (suc := successor[start]) is not None and suc < end:
            left_result = find_min_window(successor[start], end)
            if left_result and (left_result[1] - left_result[0]) < (best[1] - best[0]):
                best = left_result

        # Try sliding right boundary to the predecessor
        if predecessor[end - 1] is not None and predecessor[end - 1] >= start:
            right_result = find_min_window(start, predecessor[end - 1] + 1)
            if right_result and (right_result[1] - right_result[0]) < (
                best[1] - best[0]
            ):
                best = right_result

        _cache[(start, end)] = best
        return best

    # Find the minimum window
    window = find_min_window(0, M)
    return s[window[0] : window[1]] if window else ""


def minWindowOptimal(s: str, t: str) -> str:
    if not s or not t:
        return ""

    # Count characters needed from t
    t_counter = Counter(t)
    required = len(t_counter)  # Number of unique characters in t

    # Sliding window pointers
    left = 0
    formed = 0  # Number of unique characters in current window with desired frequency

    # Dictionary to keep count of characters in current window
    window_counts = {}

    # Result: (window length, left, right)
    result = float("inf"), None, None

    for right in range(len(s)):
        # Add character from the right to the window
        char = s[right]
        window_counts[char] = window_counts.get(char, 0) + 1

        # Check if the frequency of current character matches the desired count in t
        if char in t_counter and window_counts[char] == t_counter[char]:
            formed += 1

        print(s)
        marker = [" "] * (len(s))
        for i in range(left + 1, right + 1):
            marker[i] = "-"
        if left == right:
            marker[left] = "|"
        else:
            marker[left] = "<"
            marker[right] = ">"
        marker = "".join(marker)
        color = "\033[32m" if formed == required else "\033[31m"
        reset = "\033[0m\n"
        print(
            f"{marker} : {right - left}. {left=} {right=} {color}{s[left : right + 1]}{reset}"
        )
        print()
        # Try to contract the window until it ceases to be 'desirable'
        while left <= right and formed == required:
            char = s[left]

            # Save the smallest window so far
            if right - left + 1 < result[0]:
                result = (right - left + 1, left, right)

            # Remove character from the left of the window
            window_counts[char] -= 1
            if char in t_counter and window_counts[char] < t_counter[char]:
                formed -= 1

            # Move the left pointer ahead for the next iteration
            left += 1

    # Return the smallest window or empty string if none found
    return "" if result[0] == float("inf") else s[result[1] : result[2] + 1]  # type: ignore


if __name__ == "__main__":
    inputs = [
        ("a", "aa"),
        ("aa", "aa"),
        ("abcabcaab", "abc"),
        (
            "caacbabbbcacbabaabcbbbcbbcbbbbbbabbcacbbcbabccbabbc",
            "bababbcabccccbabbacacb",
        ),
    ]
    for s, t in inputs:
        result = minWindowOptimal(s, t)
        print(f"{result=} \n\n")
