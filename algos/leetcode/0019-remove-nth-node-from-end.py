from __future__ import annotations

from collections import deque


class ListNode:
    def __init__(self, val: int = 0, next_: ListNode | None = None):
        self.val = val
        self.next = next_

    def __eq__(self, other: ListNode | None) -> bool:
        if other is None:
            return False
        return self.val == other.val and self.next == other.next


class Solution:
    def removeNthFromEnd(self, head: ListNode | None, n: int) -> ListNode | None:
        if head is None:
            return None

        current = head
        count = 0
        while count < n and current is not None:
            count += 1
            current = current.next

        if count < n:
            return head

        hist = deque([head], maxlen=2)
        while current is not None:
            current = current.next
            hist.append(hist[-1].next)

        if hist[-1] == head:
            return head.next

        hist[-2].next = hist[-1].next
        return head


if __name__ == "__main__":

    def make_list():
        return ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5)))))

    assert Solution().removeNthFromEnd(make_list(), 2) == ListNode(
        1, ListNode(2, ListNode(3, ListNode(5)))
    )
    assert Solution().removeNthFromEnd(make_list(), 4) == ListNode(
        1, ListNode(3, ListNode(4, ListNode(5)))
    )
    assert Solution().removeNthFromEnd(make_list(), 5) == ListNode(
        2, ListNode(3, ListNode(4, ListNode(5)))
    )
    assert Solution().removeNthFromEnd(head := make_list(), 6) == head
