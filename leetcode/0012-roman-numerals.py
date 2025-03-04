from collections.abc import Sequence


class Solution:
    def get_digits_with_power(self, num: int) -> Sequence[tuple[int, int]]:
        result = []
        for idx, d in enumerate(reversed(str(num))):
            if num == 0:
                result.append((0, 1))
            else:
                result.append((int(d), idx))
        return result

    def int_to_roman(self, num: int) -> str:
        lookup_table = {
            (1, 0): "I",
            (5, 0): "V",
            (1, 1): "X",
            (5, 1): "L",
            (1, 2): "C",
            (5, 2): "D",
            (1, 3): "M",
        }
        digits_with_power = self.get_digits_with_power(num)
        result_parts = []
        for digit, power in digits_with_power:
            if digit == 4:
                result_parts.append(lookup_table[(5, power)])
                result_parts.append(lookup_table[(1, power)])
            elif digit == 9:
                result_parts.append(lookup_table[(1, power + 1)])
                result_parts.append(lookup_table[(1, power)])
            else:
                add_parts = []
                remaining_digit = digit
                if remaining_digit >= 5:
                    add_parts.append(lookup_table[(5, power)])
                    remaining_digit -= 5
                while remaining_digit != 0:
                    add_parts.append(lookup_table[(1, power)])
                    remaining_digit -= 1
                result_parts.extend(reversed(add_parts))
        return "".join(reversed(result_parts))


if __name__ == "__main__":
    solver = Solution()
    assert solver.int_to_roman(3749) == "MMMDCCXLIX"
    assert solver.int_to_roman(40) == "XL"
    assert solver.int_to_roman(600) == "DC"
    assert solver.int_to_roman(10) == "X"
    assert solver.int_to_roman(1) == "I"
