import sys
import time
from collections import Counter
from collections.abc import Collection

import pytest


def find_substring(s: str, words: list[str]) -> list[int]:
    result = []
    matches = set()
    if not s or not words:
        return []
    d = len(words[0])
    if len(s) < d * len(words):
        return []
    w = Counter(words)
    width = d * len(words)

    for i in range(len(s) - width + 1):
        window = s[i : i + width + 1]
        if window in matches:
            result.append(i)
            continue
        w_cpy = w.copy()
        for j in range(len(words)):
            if w_cpy.get((wrd := window[j * d : (j + 1) * d])):
                w_cpy[wrd] -= 1
        if w_cpy.most_common()[0][1] == 0:
            result.append(i)
            matches.add(window)

    return result


@pytest.mark.parametrize(
    "s, words, expected",
    [
        ("barfoothefoobarman", ["foo", "bar"], [0, 9]),
        ("wordgoodgoodgoodbestword", ["word", "good", "best", "word"], []),
        ("barfoofoobarthefoobarman", ["bar", "foo", "the"], [6, 9, 12]),
        ("a" * 1000, ["a"] * 500, list(range(501))),
    ],
)
def test_find_substring(s: str, words: Collection[str], expected: Collection[int]):
    start = time.perf_counter()
    result = find_substring(s, words)
    end = time.perf_counter()
    assert Counter(result) == Counter(expected)
    assert end - start < 1, f"Time limit exceeded. Took {end - start:.3e} seconds"


if __name__ == "__main__":
    pytest.main([__file__], *sys.argv[1:])
