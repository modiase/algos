"""
Given a linked list, sort it in O(n log n) time and constant space.

For example, the linked list 4 -> 1 -> -3 -> 99 should become -3 -> 1 -> 4 -> 99.


Notes
==========
C: N
T: 30,60
PD: 5

Completed after looking at example solution method, but not code,
in about 1h - 1h 30m.
I sketched a solution in one session and then implemented in another.

I looked at the solution notes but did not copy the implementation.
I had been thinking along the right lines of some sort of merge sort.
The required complexity was n log n so this was never meant to be
algorithmically difficult. The O(1) space requirement just meant not
copying the list.

Overall, the implementation was not as daunting as first assumed. Once
I had sketched out the steps that needed to happen it was surprisingly
not too hard to avoid edge cases. I was tripped up  by a few off by one
errors, but these were easy to spot and fix.

Printing the progress was very helpful and allowed me to spot deviations
in expected progress from what was actually happening.

My main take away is to take time to sketch out a solution before trying
to implement it - this is advice that should be obvious but I frequently
do not adhere to due to impatience to start tackling the problem. Sketching
a solution more than paid for itself in time in this case.

EDIT: I'm not sure my solution isn't N^2 * log (N)
"""

from __future__ import annotations

from typing import Generic, List, Optional, TypeVar


T = TypeVar("T")


class Node(Generic[T]):
    class Iterator:
        def __init__(self, head: Node[T]):
            self.head = head
            self.current = head

        def __iter__(self):
            return self

        def __next__(self):
            if self.current is None:
                raise StopIteration
            value = self.current.value
            self.current = self.current.next
            return value

    def __init__(self, value: T, next: Optional[Node[T]] = None):
        self.value = value
        self.next = next

    def __iter__(self):
        return Node.Iterator(self)

    @classmethod
    def from_list(cls, l: List[T]) -> Node[T]:
        if len(l) == 0:
            raise ValueError("List must not be empty")
        head = Node(l[0])
        node = head
        for item in l[1:]:
            nxt = Node(item)
            node.next = nxt
            node = nxt

        return head


def sort_linked_list(head: Node[int]):
    node = head
    n = 1
    offset = 0
    ptr1_offset = 0
    ptr2_offset = 0
    N = len(list(head))
    while n < N:
        for i in range(0, N // n + 1, 2):
            # 1. Set up the two pointers at their correct positions
            while offset < i * n:
                if node.next is not None:
                    offset += 1
                    node = node.next
                else:
                    break
            ptr1 = node
            ptr1_offset = offset
            while offset < (i + 1) * n:
                if node.next is not None:
                    offset += 1
                    node = node.next
                else:
                    break
            ptr2 = node
            ptr2_offset = offset
            if ptr1 == ptr2 or (ptr2_offset - ptr1_offset) < n:
                continue

            # 2. Loop : swap the pointed values if necessary; advanced first pointer
            # and if a swap occurs use the third pointer to bubble the value to the correct
            # position in the second list
            while ptr1_offset < ptr2_offset:
                if ptr2.value < ptr1.value:
                    # Swap
                    tmp = ptr1.value
                    ptr1.value = ptr2.value
                    ptr2.value = tmp

                    # Bubble
                    ptr3 = ptr2
                    ptr3_offset = ptr2_offset
                    while ptr3.next is not None and (ptr3_offset + 1 - ptr2_offset < n):
                        if ptr3.next.value < ptr3.value:
                            tmp = ptr3.next.value
                            ptr3.next.value = ptr3.value
                            ptr3.value = tmp
                        ptr3 = ptr3.next
                        ptr3_offset += 1

                ptr1 = ptr1.next
                ptr1_offset += 1

        node = head
        offset = 0
        ptr1_offset = 0
        ptr2_offset = 0
        n *= 2

    return head


l = [-16, -12, 0, 46, 96, -59, -72]
assert list(sort_linked_list(Node.from_list(l))) == [-72, -59, -16, -12, 0, 46, 96]

l = [16, 62, 45, -56, 19, -97, 74, -22, 0, -17, 23, 31, 76, -59, 53, 57]
assert list(sort_linked_list(Node.from_list(l))) == [
    -97,
    -59,
    -56,
    -22,
    -17,
    0,
    16,
    19,
    23,
    31,
    45,
    53,
    57,
    62,
    74,
    76,
]

l = [94, -24]
assert list(sort_linked_list(Node.from_list(l))) == [-24, 94]

l = [34, -20, -77, 29, -57, 7, 24, 25, -22, -78, 49, -94]
assert list(sort_linked_list(Node.from_list(l))) == [
    -94,
    -78,
    -77,
    -57,
    -22,
    -20,
    7,
    24,
    25,
    29,
    34,
    49,
]

l = [30, 91, 40, -89, -17, -93, 67, 47, 100, 51, -61, -49]
assert list(sort_linked_list(Node.from_list(l))) == [
    -93,
    -89,
    -61,
    -49,
    -17,
    30,
    40,
    47,
    51,
    67,
    91,
    100,
]

l = [98, 91, -100, -13, -89, -30, -25, 97]
assert list(sort_linked_list(Node.from_list(l))) == [
    -100,
    -89,
    -30,
    -25,
    -13,
    91,
    97,
    98,
]

l = [-9, -48, 99, -5, 7, -91, -65, 7, -7, 66, -16, -54, 49, -18]
assert list(sort_linked_list(Node.from_list(l))) == [
    -91,
    -65,
    -54,
    -48,
    -18,
    -16,
    -9,
    -7,
    -5,
    7,
    7,
    49,
    66,
    99,
]

l = [-37, 52, 86, -26, 9, 41, 42, 42, -70]
assert list(sort_linked_list(Node.from_list(l))) == [
    -70,
    -37,
    -26,
    9,
    41,
    42,
    42,
    52,
    86,
]

l = [67, -13, 23, 90, -95, -55, 4, 81, -36, 72, -1, 26, 11, -96, -40]
assert list(sort_linked_list(Node.from_list(l))) == [
    -96,
    -95,
    -55,
    -40,
    -36,
    -13,
    -1,
    4,
    11,
    23,
    26,
    67,
    72,
    81,
    90,
]

l = [97, 14, -34, -1, -39, -74, 88, 95, -86]
assert list(sort_linked_list(Node.from_list(l))) == [
    -86,
    -74,
    -39,
    -34,
    -1,
    14,
    88,
    95,
    97,
]

l = [-61, 54, -26, 34, 28, 90, -94]
assert list(sort_linked_list(Node.from_list(l))) == [-94, -61, -26, 28, 34, 54, 90]

l = [79, -98, 2, -65, 62, 8, -19]
assert list(sort_linked_list(Node.from_list(l))) == [-98, -65, -19, 2, 8, 62, 79]

l = [
    -14,
    -18,
    -18,
    55,
    36,
    -58,
    56,
    14,
    43,
    -39,
    4,
    74,
    47,
    -86,
    7,
    10,
    -3,
    77,
    -56,
    98,
]
assert list(sort_linked_list(Node.from_list(l))) == [
    -86,
    -58,
    -56,
    -39,
    -18,
    -18,
    -14,
    -3,
    4,
    7,
    10,
    14,
    36,
    43,
    47,
    55,
    56,
    74,
    77,
    98,
]

l = [-61, 2, -99, 69, -87, 24, 29, 34, -38, 93]
assert list(sort_linked_list(Node.from_list(l))) == [
    -99,
    -87,
    -61,
    -38,
    2,
    24,
    29,
    34,
    69,
    93,
]

l = [54, 5]
assert list(sort_linked_list(Node.from_list(l))) == [5, 54]

l = [54]
assert list(sort_linked_list(Node.from_list(l))) == [54]
