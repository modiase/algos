"""
Given an array of integers and a number k, where 1 <= k <= length of the array, compute
the maximum values of each subarray of length k.

For example, given array = [10, 5, 2, 7, 8, 7] and k = 3, we should get: [10, 7, 8, 8], since:

10 = max(10, 5, 2)
7 = max(5, 2, 7)
8 = max(2, 7, 8)
8 = max(7, 8, 7)

Do this in O(n) time and O(k) space. You can modify the input array in-place and
you do not need to store the results. You can simply print them out as you compute them.

Notes
===================
Completed in 45m: N

Produced solution for problem
Solution is O(n) in time
Solution is not O(k) in space but O(n) in the worst-case

range(0,10) gives space usage = N since elements outside 
the window are not removed because they are never the max.
"""
from typing import List
from heapq import heapify, heappop, heappush


def subarray_max(arr: List[int], k: int) -> List[int]:
    N = len(arr)
    subarr = [(-x, idx) for (idx, x) in enumerate(arr[:k], 0)]
    heapify(subarr)
    result = [-subarr[0][0]]

    subarr_max_size = k

    for idx, elem in enumerate(arr[k:N], k):

        subarr_max_size = max(k, len(subarr))
        heappush(subarr, (-elem, idx))
        max_item = subarr[0]

        while max_item[1] <= idx - k:
            max_item = heappop(subarr)
            if max_item[1] > idx - k:
                heappush(subarr, max_item)

        result.append(-max_item[0])

    print(k, subarr_max_size)
    return result


print(subarray_max([10, 5, 2, 7, 8, 7], 3))
print(subarray_max([10, 5, 10, 2, 7, 8, 7], 3))
print(subarray_max(list(range(0, 10)), 3))
print(subarray_max([1, 0, 0, 0, 0, 0, 0, 1], 2))
print(subarray_max([1, 0, 0, 1], 2))
