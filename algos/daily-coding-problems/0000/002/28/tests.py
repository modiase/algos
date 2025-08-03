import unittest

from .solution import justify, main


class TestCase(unittest.TestCase):
    def setUp(self):
        self.test_string = [
            "the",
            "quick",
            "brown",
            "fox",
            "jumps",
            "over",
            "the",
            "lazy",
            "dog",
        ]
        self.test_k = 16

    def test_given_example_has_correct_number_of_lines(self):
        expected_output = 3
        self.assertEqual(len(main(self.test_string, self.test_k)), expected_output)

    def test_given_example_has_correct_number_of_chars_per_line(self):
        computed_result = main(self.test_string, self.test_k)
        self.assertTrue(all([len(line) == 16 for line in computed_result]))

    def test_given_example_has_correct_number_of_spaces_on_first_line(self):
        expected_output = 3
        first_line = main(self.test_string, self.test_k)[0]
        spaces_on_first_line = sum([1 if x == " " else 0 for x in first_line])
        self.assertEqual(spaces_on_first_line, expected_output)

    def test_given_example_has_correct_number_of_spaces_on_second_line(self):
        expected_output = 4
        first_line = main(self.test_string, self.test_k)[1]
        spaces_on_first_line = sum([1 if x == " " else 0 for x in first_line])
        self.assertEqual(spaces_on_first_line, expected_output)

    def test_lines_with_only_one_word_pad_on_the_right_hand_side(self):
        expected_output = "the   "
        first_line = main(self.test_string, 6)[0]
        self.assertEqual(first_line, expected_output)

    def test_justify(self):
        test_input = "fox jumps over"
        expected_ouput = "fox  jumps  over"
        self.assertEqual(justify(test_input, 16), expected_ouput)


if __name__ == "__main__":
    unittest.main()
