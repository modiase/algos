"""
This problem was asked by Google.

An XOR linked list is a more memory efficient doubly linked list. Instead of
each node holding next and prev pointers, it holds a field named both, which
points to another node of the list.
"""

from __future__ import annotations

from collections.abc import MutableMapping

memory: MutableMapping[int, Node] = {}


class Node:
    def __init__(self, *, value: int, previous_address: int = 0):
        self.value = value
        self.both = previous_address
        memory[id(self)] = self


class XORLinkedList:
    def __init__(self, value: int):
        self.head = Node(value=value)

    def add(self, value: int) -> None:
        previous_address = 0
        current_address = id(self.head)
        current_node = memory[current_address]

        next_address = current_node.both ^ previous_address
        while next_address != 0:
            previous_address = current_address
            current_address = next_address
            current_node = memory[current_address]
            next_address = current_node.both ^ previous_address

        new_node = Node(value=value, previous_address=current_address)
        current_node.both = previous_address ^ id(new_node)

    def get(self, index: int) -> int:
        if index < 0:
            raise IndexError("Index out of range")

        previous_address = 0
        current_address = id(self.head)

        for _ in range(index):
            node = memory[current_address]
            next_address = node.both ^ previous_address

            if next_address == 0:
                raise IndexError("Index out of range")

            previous_address = current_address
            current_address = next_address

        return memory[current_address].value


if __name__ == "__main__":
    ll = XORLinkedList(0)
    for i in range(1, 20):
        ll.add(i)

    print(ll.get(0))
    print(ll.get(10))
    try:
        print(ll.get(20))
    except IndexError:
        print("Index out of range")
