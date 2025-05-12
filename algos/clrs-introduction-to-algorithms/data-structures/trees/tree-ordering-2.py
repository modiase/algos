from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterator, Iterable
from operator import itemgetter
from typing import Generic, TypeVar, Optional, Protocol


class SupportsGE(Protocol):
    @abstractmethod
    def __ge__(self, other, /) -> bool: ...


_T = TypeVar("_T", bound=SupportsGE)


def zigzag(it: Iterable[int]) -> Iterator[int]:
    for n in it:
        if n < 0:
            raise ValueError(
                f"Invalid value received '{n}'." " Value must be a natural number."
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
            f"({self.value}, {getattr(self.left, "value", None)},"
            f" {getattr(self.right, "value", None)})"
        )

    def __repr__(self):
        return (
            f"TreeNode(value={self.value}, left={repr(self.left)},"
            f" right={repr(self.right)})"
        )


def inorder(root: TreeNode[_T]) -> Iterator[_T]:
    current = root
    stack = []

    while stack or current is not None:
        while current:
            stack.append(current)
            current = current.left

        current = stack.pop()
        yield current.value
        current = current.right


def maybe_push(stack: list[TreeNode[_T]], value: Optional[TreeNode[_T]]) -> None:
    if value is None:
        return
    stack.append(value)


def preorder(root: TreeNode[_T]) -> Iterator[_T]:
    current = root
    stack = []

    while True:
        maybe_push(stack, current.left)
        maybe_push(stack, current.right)
        yield current.value

        if not stack:
            break
        current = stack.pop()


def postorder(root: TreeNode[_T]) -> Iterator[_T]:
    current = root
    stack = []
    last_visited = None

    while stack or current:
        while current is not None:
            stack.append(current)
            current = current.left

        peek = stack[-1]
        if peek.right is not None and last_visited != peek.right:
            current = peek.right
        else:
            yield peek.value
            last_visited = stack.pop()


if __name__ == "__main__":
    N = 5
    x = list(TreeNode(i) for i in range(N))
    root = x[N // 2]
    for node in itemgetter(*((i + N // 2) for i in zigzag(range(1, N))))(x):
        root.insert(node)

    print(repr(root))
    print(list(inorder(root)))
    print(list(preorder(root)))
    print(list(postorder(root)))
