import unittest

from .solution import main, tokenize, Token


class TestTokenize(unittest.TestCase):
    def test_basic(self):
        test_pattern = 'a*.d.*'
        expected_result = [Token('a',True),
                           Token('.'),
                           Token('d'),
                           Token('.',True)
                          ]

        computed_result = tokenize(test_pattern)

        self.assertEqual(expected_result,computed_result)
class TestSolution(unittest.TestCase):
    def test_given_which_is_expected_to_match_matches(self):
        computed_result = main('ra.','ray')
        self.assertTrue(computed_result)

    def test_given_which_is_expected_to_not_match_does_not_match(self):
        computed_result = main('ra.','raymond')
        self.assertFalse(computed_result)

    def test_other_given_which_is_expected_to_match_matches(self):
        computed_result = main('.*at','chat')
        self.assertTrue(computed_result)
    
    def test_complex_which_is_expected_to_match_matches(self):
        computed_result = main('.*jk.*wee*.*weq','jkhkh;llkjkljklkjljl;weeklsfdjdsfj;weq')
        self.assertTrue(computed_result)

    def test_complex_which_is_expected_not_to_match_does_not_match(self):
        computed_result = main('.*jk*.wee.*weqr','jkhkh;llkjkljklkjljl;weeklsfdjdsfj;weqtjhuafjs')
        self.assertFalse(computed_result)


if __name__ == '__main__':
    unittest.main()