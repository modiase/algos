class Solution:
    def three_some_closest(self, nums: list[int], target: int) -> int:
        sorted_nums = sorted(nums)
        N = len(nums)
        closest_sum = 2**100

        for current_idx in range(N):
            lp, rp = 0, N - 1
            while lp < rp:
                if lp == current_idx:  # skip to prevent duplication
                    lp += 1
                    continue
                if rp == current_idx:  # skip to prevent duplication
                    rp -= 1
                    continue
                triplet_sum = (
                    sorted_nums[lp] + sorted_nums[rp] + sorted_nums[current_idx]
                )
                if abs(target - triplet_sum) < abs(
                    target - closest_sum
                ):  # update closest_sum
                    closest_sum = triplet_sum

                too_high = target - triplet_sum < 0  # triplet_sum is too big
                if too_high:
                    start_r = rp
                    while sorted_nums[start_r] == sorted_nums[rp] and lp < rp:
                        rp -= 1
                else:
                    start_l = lp
                    while sorted_nums[start_l] == sorted_nums[lp] and lp < rp:
                        lp += 1

        return closest_sum


if __name__ == "__main__":
    solver = Solution()
    assert solver.three_some_closest([-1, 2, 1, -4], 1) == 1
    assert solver.three_some_closest([0, 0, 0], 1) == 1
