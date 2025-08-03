from __future__ import annotations

from typing import Generic, Optional, TypeVar

from .Node import Node

S = TypeVar("S")


class LinkedList(Generic[S]):
    def __init__(self, *args: S):
        if not args:
            return
        self.head = Node(args[0])
        self._ptr = self.head
        for arg in args[1:]:
            node = Node(arg)
            self._ptr.next = node
            self._ptr = node
        self._ptr = self.head

    def _return_next_and_advance_ptr(self) -> Optional[Node[S]]:
        if not self._ptr:
            return None
        nxt = self._ptr
        self._ptr = self._ptr.next
        return nxt

    def __iter__(self):
        return self

    def __next__(self) -> S:
        nxt = self._return_next_and_advance_ptr()
        if not nxt:
            raise StopIteration
        return nxt.payload
