from unittest import TestCase

from .solution import main


class TestSolution(TestCase):
    def setUp(self):
        self.data = [(30,75),(0,50),(60,150)]
    
    def test_solution_returns_correct_answer_for_given_data(self):
        self.assertEqual(main(self.data),2)