#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.pytest
"""
You are given an array of integers nums, there is a sliding window of size k
which is moving from the very left of the array to the very right. You can only
see the k numbers in the window. Each time the sliding window moves right by
one position.

Return the max sliding window.



Example 1:

Input: nums = [1, 3, -1, -3, 5, 3, 6, 7], k = 3
Output: [3, 3, 5, 5, 6, 7]
Explanation:
Window position                Max
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7
Example 2:

Input: nums = [1], k = 1
Output: [1]
"""

from collections import deque
import sys
from heapq import heapify, heappop, heappush
from collections.abc import Sequence

import pytest

"""
while max is out of window
  heap.pop()
heap.insert(latest)
result.insert(heap[0])

Worst case time: n heap insertions at log(n) => O(n log(n)).
Worst space: O(n) - for a monotonically increasing input sequence.
"""


class Solution:
    def maxSlidingWindow(self, nums: Sequence[int], k: int) -> Sequence[int]:
        N = len(nums)
        if k > N:
            return []

        result = []
        heap = [(-num, idx) for idx, num in enumerate(nums[:k])]
        heapify(heap)
        result.append(-heap[0][0])

        for i in range(1, N - k + 1):
            while heap and heap[0][1] < i:
                heappop(heap)
            next_idx = i + k - 1
            heappush(heap, (-nums[next_idx], next_idx))
            result.append(-heap[0][0])

        return result

    def maxSlidingWindowOptimal(self, nums: Sequence[int], k: int) -> Sequence[int]:
        N = len(nums)
        if k > N:
            return []
        result = []
        deq: deque[int] = deque([])
        for i in range(k - 1):
            while deq and nums[deq[-1]] < nums[i]:
                deq.pop()
            deq.append(i)
        for i in range(N - k + 1):
            idx = i + k - 1
            while deq and deq[0] < i:
                deq.popleft()
            while deq and nums[deq[-1]] < nums[idx]:
                deq.pop()
            deq.append(idx)
            result.append(nums[deq[0]])

        return result


@pytest.mark.parametrize(
    "inputs, expected", [(([1, 3, -1, -3, 5, 3, 6, 7], 3), [3, 3, 5, 5, 6, 7])]
)
def test_max_sliding_window(
    inputs: tuple[Sequence[int], int], expected: Sequence[int]
) -> None:
    assert Solution().maxSlidingWindow(*inputs) == expected


@pytest.mark.parametrize(
    "inputs, expected", [(([1, 3, -1, -3, 5, 3, 6, 7], 3), [3, 3, 5, 5, 6, 7])]
)
def test_max_sliding_window_optimal(
    inputs: tuple[Sequence[int], int], expected: Sequence[int]
) -> None:
    assert Solution().maxSlidingWindowOptimal(*inputs) == expected


if __name__ == "__main__":
    pytest.main([__file__, *sys.argv[1:]])
