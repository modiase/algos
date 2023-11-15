"""
Given an absolute pathname that may have . or .. as part of it,
return the shortest standardized path.

For example, given "/usr/bin/../bin/./scripts/../",
return "/usr/bin/".
Notes
======
C: 
T: 20,
PD: 
"""
import re


# class StateMachine:
#     def __init__(self, input: str):
#         self._state = input
#
#     def next(self):
#         current_state = self._state
#         while (next_state := self._next()) != current_state:
#             current_state = next_state
#
#     def _next(self):
#         current_state = self._state
#         mo = re.search(r'/\w/../', current_state):
#         if mo is not None:
#             new_state = current_state[mo.pos]
#             mo.endpos
#
#
#
#
#         return self._state


def canonical_path(path: str) -> str:
    parent_re = re.compile(r'/[^/]+?/\.\.')
    current_re = re.compile(r'(/[^/]+?)(/\.)/')

    def rep(mo):
        print(mo.group(1))
        return mo.group(2)
    print(current_re.sub(rep, path))
    return current_re.sub(lambda mo: mo.group(2), path)


assert canonical_path('/usr/bin/.././scripts/../') == '/usr/bin/'
