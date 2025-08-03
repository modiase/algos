import unittest
from solution import count_decodings


class TestSolution(unittest.TestCase):
    def test_one(self):
        """Tests the given example"""
        self.assertEqual(count_decodings("111"), 3)

    def test_two(self):
        """Test hand performed decoding"""
        self.assertEqual(count_decodings("123413"), 6)
