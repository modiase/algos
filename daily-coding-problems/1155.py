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

EDIT:
My solution is actually O(n * log n) since the heap is O(n)
in size and the insertion of a new element is therefore O(log n)
and not O(1) as was my erroneous assumption in assuming O(n)
overall complexity.
"""
from typing import List
from collections import deque
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

    return result


print(subarray_max([10, 5, 2, 7, 8, 7], 3))
print(subarray_max([10, 5, 10, 2, 7, 8, 7], 3))
print(subarray_max(list(range(0, 10)), 3))
print(subarray_max(list(range(9, 0, -1)), 3))
print(subarray_max([1, 0, 0, 0, 0, 0, 0, 1], 2))
print(subarray_max([1, 0, 0, 1], 2))


def correct_solution(arr, k):
    q = deque()
    result = []
    """
    Initialise a queue with the property that
    the index of the maximum windowed value is
    at the front (rightmost). If a larger value
    is seen LATER than a smaller one then
    all smaller values are discarded and
    only the index of the larger one is kept.
    """
    for i in range(k):
        # Remember q might be empty
        while q and arr[i] >= arr[q[-1]]:
            q.pop()
        q.append(i)

    """
    The key idea here is the loop invariants
    which are that the largest valid 
    value seen is the first value in the 
    queue and that only values which are 
    relevant for evaluation are kept to the
    right of this. Doing this ensures that
    indices are in increasing order which allows
    popping left to remove values outside the
    window.

    To maintain this invariant
    all invalid (out of window) values are first
    removed from the head of the queue at the 
    start of each iteration, and then 
    from the right the new value is compared
    with values until all that are less than it
    are removed (since they cannot be max with
    this new value in the valid window).
    Think of it as new large values bubbling up 
    from the back of the queue and removing older
    smaller values which can never be the max again.
    """
    for i in range(k, len(arr)):
        result.append(arr[q[0]])

        while q and q[0] <= i - k:
            q.popleft()

        while q and arr[i] >= arr[q[-1]]:
            q.pop()
        q.append(i)
    result.append(arr[q[0]])
    return result


print()
print(correct_solution([10, 5, 2, 7, 8, 7], 3))
print(correct_solution([10, 5, 10, 2, 7, 8, 7], 3))
print(correct_solution(list(range(0, 10)), 3))
print(correct_solution(list(range(9, 0, -1)), 3))
print(correct_solution([1, 0, 0, 0, 0, 0, 0, 1], 2))
print(correct_solution([1, 0, 0, 1], 2))
