"""
You are given a 0-indexed array of integers nums of length n. You are initially positioned at nums[0].

Each element nums[i] represents the maximum length of a forward jump from index i. In other words, if you are at nums[i], you can jump to any nums[i + j] where:

0 <= j <= nums[i] and
i + j < n
Return the minimum number of jumps to reach nums[n - 1]. The test cases are generated such that you can reach nums[n - 1].



Example 1:

Input: nums = [2,3,1,1,4]
Output: 2
Explanation: The minimum number of jumps to reach the last index is 2. Jump 1 step from index 0 to 1, then 3 steps to the last index.
Example 2:

Input: nums = [2,3,0,1,4]
Output: 2


Constraints:

1 <= nums.length <= 104
0 <= nums[i] <= 1000
It's guaranteed that you can reach nums[n - 1].

Notes
=====
## Summary
T: 30
C: Y
PD: 3

## Comments

This was  satisfying to solve because I was initially quite daunted but by taking a step back and thinking about jump game i
which I had already solved and what I knew about greedy algorithms I was able to come up with a solution.

tags: greedy-algorithms, optimisation
"""

from typing import List


class Solution:
    def jump(self, nums: List[int]) -> int:
        p = 0
        N = len(nums)
        jump_indices = []
        while p < N - 1:
            jump_indices.append(p)
            l = [
                (i, x + i)
                for (i, x) in enumerate(
                    [nums[i] for i in range(p + 1, min(p + nums[p] + 1, N))], 1
                )
            ]
            max_v = 0
            next_idx = p
            for i, v in l:
                if p + i == N - 1:
                    return len(jump_indices)
                elif max_v <= v:
                    max_v = v
                    next_idx = p + i
            p = next_idx
        jump_indices.append(N - 1)

        return len(jump_indices) - 1
