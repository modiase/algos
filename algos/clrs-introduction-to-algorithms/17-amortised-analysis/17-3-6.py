import random
from contextlib import suppress
from itertools import chain
from typing import Generic, TypeVar

T = TypeVar("T")


class TwoStackQueue(Generic[T]):
    def __init__(self):
        self._stack1: list[T] = []
        self._stack2: list[T] = []
        self._hidden: int = 0

    def enqueue(self, x: T) -> None:
        self._stack2.append(x)

    def dequeue(self) -> T:
        if not self._stack1:
            while self._stack2:
                self._stack1.append(self._stack2.pop())

        if not self._stack1:
            raise ValueError("Queue is empty")

        return self._stack1.pop()

    @property
    def potential(self) -> int:
        return len(self._stack1)

    def __iter__(self):
        return chain(reversed(self._stack1), self._stack2)


if __name__ == "__main__":
    queue = TwoStackQueue()
    queue.enqueue(1)
    queue.enqueue(2)
    queue.enqueue(3)
    assert list(queue) == [1, 2, 3], f"Expected [1, 2, 3], got {list(queue)}"
    assert (potential := queue.potential) >= 0, f"Potential is {potential}"

    queue.dequeue()
    assert list(queue) == [2, 3], f"Expected [2, 3], got {list(queue)}"
    assert (potential := queue.potential) >= 0, f"Potential is {potential}"
    queue.enqueue(4)

    assert list(queue) == [2, 3, 4], f"Expected [2, 3, 4], got {list(queue)}"
    assert (potential := queue.potential) >= 0, f"Potential is {potential}"

    queue.dequeue()

    assert list(queue) == [3, 4], f"Expected [3, 4], got {list(queue)}"
    assert (potential := queue.potential) >= 0, f"Potential is {potential}"

    for i in range(100):
        operation = random.choice(["enqueue", "dequeue"])
        match operation:
            case "enqueue":
                queue.enqueue(i)
            case "dequeue":
                with suppress(ValueError):
                    queue.dequeue()
        assert (potential := queue.potential) >= 0, f"Potential is {potential}"

    print(queue.potential)
    print(list(queue))
