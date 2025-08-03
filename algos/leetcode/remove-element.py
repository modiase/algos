"""

# Notes

## Review
T: 5
C: Y
PD: 2

## Comments

Was helpful doing the sorted array merge since it helped me
realise that you can also start from the back.



"""

from typing import List


def remove_element(nums: List[int], val: int) -> int:
    b = len(nums) - 1
    for a in range(len(nums) - 1, -1, -1):
        if nums[a] == val:
            tmp = nums[b]
            nums[b] = val
            nums[a] = tmp
            b -= 1
    return b + 1


l0 = [3, 2, 2, 3]
print(remove_element(l0, 3))
print(l0)
