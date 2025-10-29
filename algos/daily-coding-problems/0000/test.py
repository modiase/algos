import unittest
from solution import BruteForceStrategy


class TestCases(unittest.TestCase):
    def test_given_one(self):
        arr = [2, 4, 6, 2, 5]
        strategy = BruteForceStrategy()
        self.assertEqual(strategy(arr)[0], 13)

    def test_given_two(self):
        arr = [5, 1, 1, 5]
        strategy = BruteForceStrategy()
        self.assertEqual(strategy(arr)[0], 10)


if __name__ == "__main__":
    unittest.main()
