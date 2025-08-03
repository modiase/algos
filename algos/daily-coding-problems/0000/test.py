import unittest
from solution import BruteForceStrategy


class TestCases(unittest.TestCase):
    def test_given_one(self):
        l = [2, 4, 6, 2, 5]
        strategy = BruteForceStrategy()
        self.assertEqual(strategy(l)[0], 13)

    def test_given_two(self):
        l = [5, 1, 1, 5]
        strategy = BruteForceStrategy()
        self.assertEqual(strategy(l)[0], 10)


if __name__ == "__main__":
    unittest.main()
