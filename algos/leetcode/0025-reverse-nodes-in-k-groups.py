from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

import pytest


class ListNode:
    def __init__(self, val: int = 0, next: Optional[ListNode] = None):
        self.val = val
        self.next = next

    def __eq__(self, other: object) -> bool:
        """Compare two linked lists for equality."""
        if not isinstance(other, ListNode):
            return False

        current_self = self
        current_other = other

        while current_self and current_other:
            if current_self.val != current_other.val:
                return False
            current_self = current_self.next
            current_other = current_other.next

        return current_self is None and current_other is None

    def __repr__(self):
        """String representation of the linked list for debugging."""
        result = []
        current = self
        while current:
            result.append(str(current.val))
            current = current.next
        return " -> ".join(result)


class Solution:
    """My original solution."""

    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        if head is None:
            return None
        if k == 1:
            return head

        # Handle first group case
        # The 'previous group last node is None'
        # The new group head must be returned as the overall head.
        prev_node = None
        current_node = head
        for _ in range(k):
            if current_node is None:
                return head
            current_node = current_node.next
        current_node = head

        for _ in range(k):
            next_node = current_node.next
            if next_node is None:
                head.next = None
                current_node.next = prev_node
                return current_node
            current_node.next = prev_node
            prev_node = current_node
            current_node = next_node

        new_head = prev_node
        prev_group_last_node = head

        while current_node.next is not None:
            # Start iteration on first node of next group
            og_group_head = current_node
            # Reverse group and get last node in group (which is now the first)
            flag = True
            for _ in range(k):
                # Explore
                if current_node.next is None:
                    break
                current_node = current_node.next
            else:
                flag = False
            if flag:
                prev_group_last_node.next = current_node
                current_node = og_group_head
                break
            current_node = og_group_head
            for _ in range(k):
                next_node = current_node.next
                current_node.next = prev_node
                prev_node = current_node
                current_node = next_node

            # Set next_node of previous last node to new group head
            og_group_head.next = None
            prev_group_last_node.next = prev_node

            # Set current node to old group head to maintain loop invariant
            prev_group_last_node = og_group_head

        # Special case for if the last group contains exactly k items
        og_group_head = current_node
        for i in range(k - 1):
            if current_node.next is None:
                # Fewer than k - do nothing
                current_node = og_group_head
                break
            current_node = current_node.next
        else:
            prev_node = None
            current_node = og_group_head
            for _ in range(k):
                next_node = current_node.next
                current_node.next = prev_node
                prev_node = current_node
                current_node = next_node
            current_node = prev_node
        prev_group_last_node.next = current_node
        return new_head


class ClaudeSolution:
    """Claude 4.0's solution."""

    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        if not head or k == 1:
            return head

        def try_reverse_k(head, k):
            """
            Try to reverse k nodes starting from head.
            Returns (new_head, new_tail, next_start) if successful,
            or (head, None, None) if insufficient nodes.
            """
            # First check if we have k nodes
            curr = head
            for _ in range(k):
                if not curr:
                    return head, None, None  # Not enough nodes
                curr = curr.next

            # We have k nodes, reverse them
            prev = curr  # This becomes the next pointer after the reversed group
            curr = head
            tail = head  # Original head becomes new tail

            for _ in range(k):
                next_node = curr.next
                curr.next = prev
                prev = curr
                curr = next_node

            return prev, tail, curr  # new_head, new_tail, next_start

        # Try to reverse first group
        new_head, prev_tail, curr = try_reverse_k(head, k)
        if not prev_tail:  # First group couldn't be reversed
            return head

        # Process remaining groups
        while curr:
            group_head, group_tail, next_curr = try_reverse_k(curr, k)
            if not group_tail:
                # Couldn't reverse this group, attach remaining nodes as-is
                prev_tail.next = curr
                break
            prev_tail.next = group_head
            prev_tail = group_tail
            curr = next_curr

        return new_head


@dataclass(frozen=True, kw_only=True)
class TestCase:
    expected: ListNode
    head: ListNode
    k: int
    name: str


@pytest.mark.parametrize(
    "solution_class",
    [Solution, ClaudeSolution],
    ids=lambda cls: cls.__name__,
)
@pytest.mark.parametrize(
    "test_case_factory",
    [
        lambda: TestCase(
            name="empty list",
            head=None,
            k=1,
            expected=None,
        ),
        lambda: TestCase(
            name="k=2 with 5 nodes",
            head=ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5))))),
            k=2,
            expected=ListNode(2, ListNode(1, ListNode(4, ListNode(3, ListNode(5))))),
        ),
        lambda: TestCase(
            name="k=3 with 5 nodes",
            head=ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5))))),
            k=3,
            expected=ListNode(3, ListNode(2, ListNode(1, ListNode(4, ListNode(5))))),
        ),
        lambda: TestCase(
            name="single node with k=1",
            head=ListNode(1),
            k=1,
            expected=ListNode(1),
        ),
        lambda: TestCase(
            name="k=4 with 8 nodes",
            head=ListNode(
                1,
                ListNode(
                    2,
                    ListNode(
                        3,
                        ListNode(4, ListNode(5, ListNode(6, ListNode(7, ListNode(8))))),
                    ),
                ),
            ),
            k=4,
            expected=ListNode(
                4,
                ListNode(
                    3,
                    ListNode(
                        2,
                        ListNode(1, ListNode(8, ListNode(7, ListNode(6, ListNode(5))))),
                    ),
                ),
            ),
        ),
    ],
    ids=lambda test_case: test_case().name,
)
def test_reverse_k_group(solution_class, test_case_factory: Callable[[], TestCase]):
    """Test reversing nodes in k-groups."""
    test_case = test_case_factory()
    assert (
        solution_class().reverseKGroup(test_case.head, test_case.k)
        == test_case.expected
    )


if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
