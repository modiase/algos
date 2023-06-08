
import unittest

from .solution import main
from .structs import LinkedList


class SolutionUnitTests(unittest.TestCase):
    def setUp(self):
        self.test_data = LinkedList(*[1,2,3,4])

    def test_given_and_integer_k_and_a_linked_list_solution_removes_kth_element(self):
        expected_result = [1,2,4]
        computed_result = list(main(self.test_data,2))
        self.assertEqual(expected_result,computed_result)
        


if __name__ == '__main__':
    unittest.main()