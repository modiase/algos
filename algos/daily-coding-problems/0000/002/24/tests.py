import unittest

from .solution import Node


class Tests(unittest.TestCase):
    def setUp(self):
        root = Node(5)
        root.left = Node(10)
        root.left.left = Node(15)
        root.left.right = Node(20)
        self.root = root

    def test_dfs(self):
        computed_result = [n.payload for n in self.root]
        expected_result = [10, 15, 20]
        self.assertEqual(computed_result, expected_result)

    def test_locking_suceeds_when_children_are_unlocked(self):
        root = Node(5)
        root.left = Node(10)
        root.right = Node(20)

        computed_result = root.lock()

        self.assertTrue(computed_result)

    def test_locking_fails_when_one_child_is_locked(self):
        root = Node(5)
        root.left = Node(10)
        root.right = Node(20)
        root.left.left = Node(30)

        root.left.left.lock()
        computed_result = root.lock()

        self.assertFalse(computed_result)

    def test_locking_fails_when_many_children_are_locked(self):
        root = Node(5)
        root.left = Node(10)
        root.right = Node(20)
        root.left.left = Node(30)

        root.left.left.lock()
        root.right.lock()
        computed_result = root.lock()

        self.assertFalse(computed_result)

    def test_unlocking_suceeds_when_children_are_unlocked(self):
        root = Node(5)
        root.left = Node(10)
        root.right = Node(20)

        root.lock()
        computed_result = root.unlock()

        self.assertTrue(computed_result)

    def test_unlocking_fails_when_one_child_is_unlocked(self):
        root = Node(5)
        root.left = Node(10)
        root.right = Node(20)
        root.left.left = Node(30)

        root.left.left.lock()
        computed_result = root.unlock()

        self.assertFalse(computed_result)

    def test_unlocking_fails_when_many_children_are_unlocked(self):
        root = Node(5)
        root.left = Node(10)
        root.right = Node(20)
        root.left.left = Node(30)

        root.left.left.lock()
        root.right.lock()
        computed_result = root.unlock()

        self.assertFalse(computed_result)


if __name__ == '__main__':
    unittest.main()
