from operator import itemgetter


def fourSum(nums: list[int], target: int) -> list[list[int]]:
    nums.sort()
    N = len(nums)

    def _it():
        for i in range(N - 2):
            if i > 0 and nums[i - 1] == nums[i]:
                continue

            for j in range(i + 1, N - 1):
                if j > i + 1 and nums[j - 1] == nums[j]:
                    continue

                left = j + 1
                r = N - 1
                while left < r:
                    tot = sum(ss := itemgetter(i, j, left, r)(nums))
                    if tot == target:
                        yield ss

                    if tot >= target:
                        r -= 1
                        while r > left and nums[r + 1] == nums[r]:
                            r -= 1
                    if tot <= target:
                        left += 1
                        while left < r and nums[left - 1] == nums[left]:
                            left += 1

    return list(list(t) for t in _it())


if __name__ == "__main__":
    # Test cases
    test_cases = [
        ([1, 0, -1, 0, -2, 2], 0),  # Basic case with zero target
        ([2, 2, 2, 2, 2], 8),  # All same numbers
        ([-3, -2, -1, 0, 0, 1, 2, 3], 0),  # Sorted array with zero target
        ([1, 1, 1, 1, 1, 1, 1, 1], 4),  # Multiple same numbers
        ([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], 0),  # Large sorted array
        ([0, 0, 0, 0], 0),  # All zeros
        (
            [1000000000, 1000000000, 1000000000, 1000000000],
            4000000000,
        ),  # Large numbers
        ([-1, -5, -5, -3, 2, 5, 0, 4], -7),  # Negative target
        ([-2, -1, -1, 1, 1, 2, 2], 0),  # Mixed positive/negative
        ([0, 1, 2, 3, 4, 5], 10),  # Sequential numbers
    ]

    for nums, target in test_cases:
        result = fourSum(nums, target)
        assert all(sum(r) == target for r in result)  # check sum
        assert len(result) == len(set(tuple(sorted(r)) for r in result))  # check unique
