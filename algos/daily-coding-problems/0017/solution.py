import re
from sys import argv
from typing import List, Optional, Tuple

REGEX = re.compile(r"((?:\\t)*)(.+?)(\\n)(.*)")


def deepest_paths(tokens: List[Tuple[int, str]]) -> List[str]:
    if not tokens:
        return []
    if len(tokens) == 1:
        return tokens
    store = []
    while len(tokens) != 1:
        buffer = []
        last = tokens.pop()
        buffer.append(last[1])
        n = last[0]
        while n != 1:
            last = tokens.pop()
            if last[0] == n - 1:
                buffer = [last[1] + "/" + x for x in buffer]
                n = last[0]
            elif last[0] == n:
                buffer.append(last[1])
            else:
                raise ValueError("Input tokens are invalid.")
        store.append(buffer)
    assert len(tokens) == 1
    res = []
    store = [item for path_list in store for item in path_list]
    for item in store:
        res.append(tokens[0][1] + "/" + item)
    return res


def consume_next(s: str) -> Tuple[Optional[Tuple[int, str]], str]:
    if not s:
        return (None, "")
    mo = REGEX.match(s)
    if not mo:
        tabs = re.search(r"^((?:\\t)*)", s)[0]
        level = len(tabs) / 2 if tabs else 0
        return ((level, s.replace("\\t", "")), "")
    tabs = mo.group(1)
    level = len(tabs) / 2 if tabs else 0
    name = mo.group(2)
    tail = mo.group(4)
    return ((level, name), tail)


def main(s):
    tokens_list = []
    remainder = s
    while remainder:
        t = consume_next(remainder)
        if t[0]:
            tokens_list.append(t[0])
        remainder = t[1]
    paths = deepest_paths(tokens_list)
    lens = [len(x) for x in paths]
    max_len = max(lens)
    longest = paths[lens.index(max_len)]
    return (longest, max_len)


if __name__ == "__main__":
    s = argv[1]
    res = main(s)
    print(res)
