"""
Determine whether a doubly linked list is a palindrome.
What if it’s singly linked?

For example, 1 -> 4 -> 3 -> 4 -> 1 returns True
while 1 -> 4 returns False.

Notes:
======
C: Y
T: 5
PD: 1

Loop forward then backward in the doubly linked case.

In the singly linked case, build a reverse linked list one
element at a time as you step through forward and then loop
through that to get the reverse list. Compare the two as before.
For O(1) space complexity, you'd need to modify the list in place.
Keep a pointer to the previous node and the node before that. After
accessing the next node, set the next pointer of the previous node
to the node before it. Watch out for final pair of nodes.
"""
from __future__ import annotations

from typing import Generic, List, Optional, TypeVar

T = TypeVar('T')


class DoublyLinkedNode(Generic[T]):
    def __init__(self, val: T, next: Optional[DoublyLinkedNode[T]] = None, prev: Optional[DoublyLinkedNode[T]] = None):
        self.val = val
        self.next = next
        self.prev = prev

    @classmethod
    def from_list(cls, l: List[T]) -> DoublyLinkedNode[T]:
        if len(l) == 0:
            raise ValueError('List cannot be empty')
        head = cls(l[0])
        n0 = head
        for elem in l[1:]:
            n1 = cls(elem)
            n1.prev = n0
            n0.next = n1
            n0 = n1
        return head


def is_palindrome(head: DoublyLinkedNode[int]):
    forward = [head.val]
    node = head
    while (next := node.next) is not None:
        forward.append(next.val)
        node = next

    backward = [node.val]
    while (prev := node.prev) is not None:
        backward.append(prev.val)
        node = prev

    return forward == backward


l0 = DoublyLinkedNode.from_list([1, 4, 3, 4, 1])
l1 = DoublyLinkedNode.from_list([1, 4])

assert is_palindrome(l0)
assert not is_palindrome((l1))
