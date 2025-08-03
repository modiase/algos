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
    stack = []
    parts = re.split("[/]", path)
    for part in parts:
        if part == ".":
            continue
        elif part == ".." and stack:
            stack.pop()
        else:
            stack.append(part)

    if not stack or stack == [""]:
        return "/"
    if stack[0] != "":
        stack = [""] + stack
    return "/".join(stack)


assert canonical_path("/usr/bin/../bin/./scripts/../") == "/usr/bin/"
assert canonical_path("/../../../") == "/"
assert canonical_path("/usr/bin/env/../env/./../../") == "/usr/"
assert canonical_path("/../../../usr/bin/env/") == "/usr/bin/env/"
