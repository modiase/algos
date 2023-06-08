import unittest

import solution


class Test(unittest.TestCase):
    def test_one(self):
        """Basic test with negative number"""
        l = [3, 4, -1, 1]
        self.assertEqual(solution.lowest_missing_int(l), 2)

    def test_two(self):
        """Basic test with zero"""
        l = [1, 2, 0]
        self.assertEqual(solution.lowest_missing_int(l), 3)

    def test_three(self):
        """Basic test with duplicates"""
        l = [1, 1, 5, 6, 2, 4, 5, 6, 7, 5, 5]
        self.assertEqual(solution.lowest_missing_int(l), 3)
