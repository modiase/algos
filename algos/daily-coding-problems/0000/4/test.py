import unittest

import solution


class Test(unittest.TestCase):
    def test_one(self):
        """Basic test with negative number"""
        arr = [3, 4, -1, 1]
        self.assertEqual(solution.lowest_missing_int(arr), 2)

    def test_two(self):
        """Basic test with zero"""
        arr = [1, 2, 0]
        self.assertEqual(solution.lowest_missing_int(arr), 3)

    def test_three(self):
        """Basic test with duplicates"""
        arr = [1, 1, 5, 6, 2, 4, 5, 6, 7, 5, 5]
        self.assertEqual(solution.lowest_missing_int(arr), 3)
