import unittest

from .solution import main

class TestSolution(unittest.TestCase):

    def test_balanced_brackets_returns_true(self):
        test_input = '([])[]({})'
        computed_result = main(test_input)
        self.assertTrue(computed_result)

    def test_malformed_brackets_returns_false(self):
        test_input = '([)]'
        computed_result = main(test_input)
        self.assertFalse(computed_result)

    def test_unbalanced_brackets_returns_false(self):
        test_input = '((()'
        computed_result = main(test_input)
        self.assertFalse(computed_result)

    def test_complex_unbalanced_brackets_returns_false(self):
        test_input = '(((){[({}])}))'
        computed_result = main(test_input)
        self.assertFalse(computed_result)





if __name__ == '__main__':
    unittest.main()

