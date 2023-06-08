import unittest
from solution import car, cdr
from given import cons


class TestCase(unittest.TestCase):
    def test_car(self):
        self.assertEqual(car(cons(3, 4,)), 3)

    def test_cdr(self):
        self.assertEqual(cdr(cons(3, 4,)), 4)
