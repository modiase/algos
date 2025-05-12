from __future__ import annotations

import os
import random as rn
import time
from functools import reduce
from typing import Final, Self


# Setup
rn.seed(seed := int(os.getenv("SEED", time.time())))
print(f"{seed=}")


class BaseNode:
    def __init__(self, next_: Node | None = None):
        self.next = next_

    def successor(self, next_: Self | None = None) -> Self:
        self.next = next_ if next_ is not None else self.__class__()
        return self.next


NIL: Final = BaseNode()


class Node(BaseNode):
    def __init__(self, next_: Node | None = None):
        self.next = next_ if next_ is not None else NIL

    def __len__(self) -> int:
        current = self
        count = 0
        while current != NIL:
            if current.next is None:
                raise RuntimeError()
            current = current.next
            count += 1
        return count


l1, l2 = rn.randint(5, 10), rn.randint(5, 10)
intersection = rn.randint(1, min(l1, l2) - 1)

list1, list2 = Node(), Node()

tail1 = reduce(lambda node, _: node.successor(), range(l1 - intersection - 1), list1)
tail2 = reduce(lambda node, _: node.successor(), range(l2 - intersection - 1), list2)

for i in range(intersection):
    node = Node()
    tail1 = tail1.successor(node)
    tail2 = tail2.successor(node)


# Intersection algorithm
p1 = list1
p2 = list2

a, b = (p2, p1) if l2 > l1 else (p1, p2)
for i in range(abs(l2 - l1)):
    a = a.next

while a != NIL and b != NIL and a != b:
    if a is None or b is None or a.next is None or b.next is None:
        raise RuntimeError()
    a = a.next
    b = b.next
    print(a, b)

if a != b:
    raise RuntimeError("Could not find intersection")

print(l1, l2, intersection)
print("Intersection at ", a, b)
