from __future__ import annotations
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Generic, Iterator, Optional, Type, TypeVar


T = TypeVar('T')

class TraversalStrategy(ABC):
    def __init__(self,node : Node[Any]):
        self.node = node
    
    
    def __iter__(self)-> Iterator[Node[Any]]:
        return self

    @abstractmethod
    def __next__(self) -> Node[Any]:
        ...
    
    @abstractmethod
    def update(self):
        ...


class BreadthFirstSearch(TraversalStrategy):
    def __init__(self, node : Node[Any]):
        ...
    

class DepthFirstSearch(TraversalStrategy):
    def __init__(self, node : Node[Any]):
        super().__init__(node)
        self._nodes = deque()
        if left := self.node.left:
            self._nodes.append(left)
        if right := self.node.right:
            self._nodes.append(right)
   
    def update(self):
        self._nodes.clear()
        if left := self.node.left:
            self._nodes.append(left)
        if right := self.node.right:
            self._nodes.append(right)
    
    def __next__(self) -> Node[Any]:
        if not self._nodes:
            raise StopIteration
        head = self._nodes.popleft()
        if right := head.right:
            self._nodes.appendleft(right)
        if left := head.left:
            self._nodes.appendleft(left)
        return head
        


class Node(Generic[T]):
    def __init__(self, payload : T, left : Optional[Node[Any]] = None, right : Optional[Node[Any]] = None,
                 traversal_strategy : Type[TraversalStrategy] = DepthFirstSearch):
        self.payload = payload
        self._left = left
        self._right = right
        self.traversal_strategy = traversal_strategy(self)
        self._is_locked = False
    
    @property
    def left(self) -> Optional[Node[Any]]:
        return self._left
    
    @left.setter
    def left(self, left : Any):
        self._left = left
        self.traversal_strategy.update() # Could use subscriber pattern

    @property
    def right(self) -> Optional[Node[Any]]:
        return self._right
    
    @right.setter
    def right(self, right : Any):
        self._right = right
        self.traversal_strategy.update() # Could use subscriber pattern
    
    @property
    def is_locked(self) -> bool:
        return self._is_locked
        
    def lock(self) -> bool:
        if self._is_locked or any([ n.is_locked for n in self]):
            return False
        self._is_locked = True
        return True
    
    def unlock(self) -> bool:
        if not self._is_locked or any([ n.is_locked for n in self]):
            return False
        self._is_locked = False
        return True
        
        
    
    def __iter__(self) -> Iterator[Node[Any]]:
        return self.traversal_strategy
     
    def __str__(self):
        return f'<Node ({self.payload})>'
    
    def __repr__(self):
        return f'Node(payload={self.payload},left={self.left},right={self.right},\
            traversal_strategy={repr(self.traversal_strategy)})'
        
        


