import unittest

from .solution import main


class SolutionTestCase(unittest.TestCase):
    def test_first_given_example(self):
        test_input = [2, 1, 2]

        expected_result = 1
        computed_result = main(test_input)
        self.assertEqual(expected_result, computed_result)

    def test_second_given_example(self):
        test_input = [3, 0, 1, 3, 0, 5]

        expected_result = 8
        computed_result = main(test_input)
        self.assertEqual(expected_result, computed_result)

    def test_hand_computed_example(self):
        test_input = [3, 2, 1, 0, 1, 2, 3, 2, 1, 2, 3, 4, 3, 2, 1, 2]

        expected_result = 14
        computed_result = main(test_input)
        self.assertEqual(expected_result, computed_result)


if __name__ == '__main__':
    unittest.main()
