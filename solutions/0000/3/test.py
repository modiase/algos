from unittest import TestCase
import unittest

from given import Node
from solution import serialize, deserialize, node_regex


class Test(TestCase):

    def test_serialize(self):
        """Test the serialization method"""
        node1 = Node('val1', 'left', 'right')
        # node2 = Node('val2', Node('left'), Node('right'))
        # node3 = Node('val3', Node('left', left=None, right='left.right'), Node(
        # 'right', Node('right.left'), Node('right.right')))
        self.assertEqual(serialize(node1),
                         "(0,val1)::(1,left)::(2,right)")

    @ unittest.skip("Would have to implement a __eq__ method on Node.")
    def test_deserialize(self):
        """Test the deserialization method"""
        node = Node('val3', Node('left', left=None, right='left.right'), Node(
            'right', Node('right.left'), Node('right.right')))
        node_str = "[(val3),[(left),(),(left.right)],[(right),[(right.left),(),()],[(right.right),(),()]]]"
        raise NotImplementedError("Node requires __eq__ method.")

    def test_solution(self):
        """Tests that give n a node object, the deserialize and serialize commands when chained return the original object"""
        node = Node('root', Node('left', Node('left.left')), Node('right'))
        self.assertEqual(deserialize(
            serialize(node)).left.left.val, 'left.left')
