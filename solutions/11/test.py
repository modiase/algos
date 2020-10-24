import unittest

from solution import main


class Tests(unittest.TestCase):
    def test_given(self):
        strings = ['deer', 'deal', 'dog']
        t = 'de'
        self.assertCountEqual(main(t, strings), ['deer', 'deal'])
