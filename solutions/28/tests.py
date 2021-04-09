import unittest

from .solution import main

class UnitTests(unittest.TestCase):
    def setUp(self):
        self.test_string = ["the", "quick", "brown", "fox", 
        "jumps", "over", "the", "lazy", "dog"]
        self.test_k = 16

    def test_given_example_has_correct_number_of_lines(self):
        expected_output = 3
        self.assertEqual(len(main(self.test_string,self.test_k)),expected_output)

    def test_given_example_has_correct_number_of_chars_per_line(self):
        self.assertTrue(all([len(x) == 16 for x in main(self.test_string,
        self.test_k)]))

    def test_given_example_has_correct_number_of_spaces_on_first_line(self):
        expected_output = 3
        first_line = main(self.test_string,self.test_k)[0]
        spaces_on_first_line = sum([1 if x == ' ' else 0 for x in first_line])
        self.assertEqual(spaces_on_first_line,expected_output)

    def test_given_example_has_correct_number_of_spaces_on_second_line(self):
        expected_output = 4
        first_line = main(self.test_string,self.test_k)[1]
        spaces_on_first_line = sum([1 if x == ' ' else 0 for x in first_line])
        self.assertEqual(spaces_on_first_line,expected_output)

    def test_lines_with_only_one_word_pad_on_the_right_hand_side(self):
        expected_output = 'the   '
        first_line = main(self.test_string,6)[1]
        self.assertEqual(first_line,expected_output)



if __name__ == '__main__':
    unittest.main()