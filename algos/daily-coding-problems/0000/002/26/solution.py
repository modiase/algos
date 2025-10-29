from typing import TypeVar

from .structs import LinkedList

T = TypeVar("T")


def main(linked_list: LinkedList[T], k: int) -> LinkedList[T]:
    window_pointers = (None, None)
    head = linked_list.head
    next_node = head
    for _ in range(0, k):
        next_node = next_node.next
        window_pointers = (None, next_node)
    window_pointers = (linked_list.head, next_node)

    while window_pointers[1].next:
        window_pointers = (window_pointers[0].next, window_pointers[1].next)

    kth_element = window_pointers[0].next
    element_after_kth_element = kth_element.next
    window_pointers[0].next = element_after_kth_element
    return linked_list
