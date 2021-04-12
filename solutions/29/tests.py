import unittest

from .solution import main


class TestCase(unittest.TestCase):

    def test_given_example_returns_expected_encoding(self):
        expected_result = "4A3B2C1D2A"
        test_input = "AAAABBBCCDAA"
        self.assertEqual(expected_result, main(test_input))


if __name__ == '__main__':
    unittest.main()
