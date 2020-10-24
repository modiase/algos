import unittest
from solution import Node, count_univals


class TestSolution(unittest.TestCase):
    def test_given(self):
        root = Node(0)
        root.left = Node(1)
        root.right = Node(0)
        root.right.right = Node(0)
        root.right.left = Node(1)
        root.right.left.left = Node(1)
        root.right.left.right = Node(1)
        self.assertEqual(count_univals(root), 5)
