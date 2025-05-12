from typing import List


def euclid(b, a):
    while b:
        a, b = b, a % b
    return a


def rotate(nums: List[int], k: int) -> None:
    """
    Do not return anything, modify nums in-place instead.
    """

    N = len(nums)
    if k == 0 or N == 0:
        return

    gcd = euclid(N, k)
    r = N / gcd

    for j in range(gcd):
        jumps = 0
        i = j
        val = nums[i]
        while jumps <= r:
            i_next = i + k
            if i_next >= N:
                i_next %= N
            tmp = nums[i_next]
            nums[i_next] = val
            val = tmp

            i = i_next
            jumps += 1


def make_test_case(nums, k, expected):
    nums_cpy = nums[:]
    rotate(nums_cpy, k)
    assert nums_cpy == expected


def test_case_one():
    make_test_case([1, 2, 3, 4, 5, 6, 7], 3, [5, 6, 7, 1, 2, 3, 4])


def test_case_two():
    make_test_case([-1, 100, 3, 99], 2, [3, 99, -1, 100])


def test_case_three():
    make_test_case([1, 2, 3, 4, 5, 6], 2, [5, 6, 1, 2, 3, 4])


def test_case_four():
    make_test_case([1, 2, 3, 4, 5, 6], 2, [5, 6, 1, 2, 3, 4])


def test_case_five():
    make_test_case([1, 2, 3, 4, 5, 6], 0, [1, 2, 3, 4, 5, 6])


def test_case_six():
    make_test_case([1, 2, 3, 4, 5, 6], 6, [1, 2, 3, 4, 5, 6])


def test_case_seven():
    make_test_case([1, 2], 1, [2, 1])


def test_case_eight():
    make_test_case([], 5, [])
