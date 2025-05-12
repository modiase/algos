from collections import Counter


def fourSumCount(
    nums1: list[int], nums2: list[int], nums3: list[int], nums4: list[int]
) -> int:
    """
    We count all the possible sums of nums1 and nums2 and store them in a counter.
    Then we iterate over all the possible sums of nums3 and nums4 and check if the
    negative of the sum exists in the counter. If it does, we add the value of the
    counter to the result.

    Overall complexity is O(n^2) because we iterate over all the possible sums of
    nums3 and nums4 and check if the negative of the sum exists in the counter.

    This generalises to k-sum problems where we can iterate through k/2-tuples and then find
    the number of complements for the other k/2-tuple.
    """
    cnt = Counter(a + b for a in nums1 for b in nums2)
    return sum(cnt[-(c + d)] for c in nums3 for d in nums4)


if __name__ == "__main__":
    test_cases = [
        (([1, 2], [-2, -1], [-1, 2], [0, 2]), 2),
        (([0], [0], [0], [0]), 1),
        (([-1, -1], [-1, 1], [-1, 1], [1, -1]), 6),
        (([], [-1, 1], [-1, 1], [1, -1]), 0),
    ]
    for input, expected in test_cases:
        assert fourSumCount(*input) == expected
