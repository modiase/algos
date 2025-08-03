from __future__ import annotations

from typing import Generic, Optional, TypeVar

T = TypeVar("T")


class Node(Generic[T]):
    def __init__(self, payload: T, nextNode: Optional[Node[T]] = None):
        self.payload = payload
        self.next = nextNode
