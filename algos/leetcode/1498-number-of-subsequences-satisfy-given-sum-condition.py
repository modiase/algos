"""
After about an hour I was very close and used claude to work out the final
details.
```python
from bisect import bisect_right

class Solution:
    def numSubseq(self, nums: List[int], target: int) -> int:
        nums = sorted(nums)
        count = 0
        N = len(nums)
        for i in range(N):
            t = target-nums[i]
            r = bisect_right(nums[i:], t)
            w = r - i
            if r==N
                w -= 1
            if r < i:
                break
            print(i, r, w)
            count += int(pow(2,w))


        return count % (int(1e9) + 7)
```
"""

from bisect import bisect_right


class Solution:
    def numSubseq(self, nums: list[int], target: int) -> int:
        nums.sort()
        count = 0
        N = len(nums)
        MOD = 10**9 + 7

        for i in range(N):
            max_val = target - nums[i]
            j = bisect_right(nums, max_val) - 1

            if j >= i:
                count = (count + pow(2, j - i, MOD)) % MOD
            else:
                break

        return count
