from __future__ import annotations

from abc import abstractmethod
from operator import itemgetter
from collections.abc import Iterator, Iterable
from typing import Generic, TypeVar, Protocol


class SupportsGE(Protocol):
    @abstractmethod
    def __ge__(self, other, /) -> bool: ...


_T = TypeVar("_T", bound=SupportsGE)


def zigzag(it: Iterable[int]) -> Iterator[int]:
    for n in it:
        if n < 0:
            raise ValueError(
                f"Invalid value received '{n}'. Value must be a natural number."
            )
        if n == 0:
            yield 0
        else:
            yield ((-1) ** n) * (((n - 1) // 2) + 1)


class TreeNode(Generic[_T]):
    def __init__(
        self,
        value: _T,
        left: TreeNode[_T] | None = None,
        right: TreeNode[_T] | None = None,
    ):
        self.value = value
        self.left = left
        self.right = right

    def insert(self, node: TreeNode[_T]) -> None:
        if node.value >= self.value:
            if self.right is None:
                self.right = node
                return
            self.right.insert(node)
            return
        if self.left is None:
            self.left = node
            return
        self.left.insert(node)

    def __str__(self):
        return (
            f"({self.value}, {getattr(self.left, 'value', None)},"
            f" {getattr(self.right, 'value', None)})"
        )

    def __repr__(self):
        return (
            f"TreeNode(value={self.value}, left={repr(self.left)},"
            f" right={repr(self.right)})"
        )


def inorder(root: TreeNode[_T]) -> Iterator[_T]:
    current = root
    stack = []

    while True:
        if (right := current.right) is not None:
            stack.append((0, right))
        if current.left is None:
            break
        stack.append((1, current))
        current = current.left
    yield current.value

    while stack:
        visited, current = stack.pop()
        if visited:
            yield current.value
            continue
        while True:
            if (right := current.right) is not None:
                stack.append((0, right))
            if current.left is None:
                break
            stack.append((1, current))
            current = current.left
        yield current.value


def preorder(root: TreeNode[_T]) -> Iterator[_T]:
    current = root
    stack = []

    while True:
        if (right := current.right) is not None:
            stack.append(right)
        yield current.value
        if current.left is None:
            break
        current = current.left

    while stack:
        current = stack.pop()
        while True:
            if (right := current.right) is not None:
                stack.append(right)
            yield current.value
            if current.left is None:
                break
            current = current.left


def postorder(root: TreeNode[_T]) -> Iterator[_T]:
    current = root
    stack = []

    while current is not None:
        stack.append((1, current))
        if (right := current.right) is not None:
            stack.append((0, right))
        if current.left is None:
            break
        current = current.left

    while stack:
        visited, current = stack.pop()
        if visited:
            yield current.value
            continue
        while current is not None:
            stack.append((1, current))
            if (right := current.right) is not None:
                stack.append((0, right))
            if current.left is None:
                break
            current = current.left


if __name__ == "__main__":
    N = 5
    x = list(TreeNode(i) for i in range(N))
    root = x[N // 2]
    for node in itemgetter(*((i + N // 2) for i in zigzag(range(N))))(x):
        root.insert(node)

    print(repr(root))
    print(list(inorder(root)))
    print(list(preorder(root)))
    print(list(postorder(root)))
