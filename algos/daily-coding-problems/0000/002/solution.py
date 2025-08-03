from typing import *


class Node:
    __slots__ = "payload", "next"

    def __init__(self, payload: Any = None, nxt=None):
        self.payload = payload
        self.next = nxt


class LinkedList:
    __slots__ = "head"

    def __init__(self, head: Node):
        self.head = head

    def __iter__(self):
        nxt = self.head
        while nxt:
            yield nxt
            nxt = nxt.next


def reverse_linked_list(ll: LinkedList):
    stack = []
    for n in ll:
        stack.append(n)
    head = Node(stack.pop().payload)
    nxt = Node(stack.pop().payload)
    head.next = nxt
    while len(stack) > 0:
        tmp = Node(stack.pop().payload)
        nxt.next = tmp
        nxt = tmp
    return LinkedList(head)


def read_args(): ...


def main(args: Dict[str, Any]):
    r1 = reverse_linked_list(args["l1"])
    r2 = reverse_linked_list(args["l2"])

    ptr1 = r1.head
    ptr2 = r2.head
    while 1:
        prev1 = ptr1
        ptr1 = ptr1.next
        ptr2 = ptr2.next
        if ptr1.payload != ptr2.payload:
            return prev1.payload


if __name__ == "__main__":
    n10 = Node(10)
    n8 = Node(8, n10)
    n1 = Node(1, n8)
    n99 = Node(99, n1)
    n7 = Node(7, n8)
    n3 = Node(3, n7)

    l1 = LinkedList(n3)
    l2 = LinkedList(n99)
    main({"l1": l1, "l2": l2})
