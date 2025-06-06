from collections import defaultdict
from collections.abc import Collection

import pytest


class Trie:
    SENTINEL = -1

    class Node:
        def __init__(self):
            self._edges = defaultdict(Trie.Node)

        def __str__(self) -> str:
            return str({k: str(v) for k, v in self._edges.items()})

    def __init__(self):
        self._root = self.Node()

    def insert(self, word: str) -> None:
        current = self._root
        for c in word:
            next_node = current._edges[c]
            current = next_node
        next_node = current._edges[Trie.SENTINEL]

    def _search(self, word: str) -> Node | None:
        current = self._root
        for c in word:
            current = current._edges.get(c)
            if current is None:
                return
        return current

    def search(self, word: str) -> bool:
        return ((node := self._search(word)) is not None) and (
            node._edges.get(Trie.SENTINEL) is not None
        )

    def prefix(self, prefix: str) -> bool:
        return self._search(prefix) is not None

    @classmethod
    def of(cls, words: Collection[str]) -> "Trie":
        t = cls()
        for word in words:
            t.insert(word)
        return t

    def __str__(self) -> str:
        return str(self._root)


@pytest.mark.parametrize(
    "words,search_word,expected",
    [
        (["abc"], "abc", True),
        (["abc"], "def", False),
        (["abc"], "ab", False),
        (["abc"], "a", False),
        (["abc"], "abcd", False),
        (["abc", "ab"], "abc", True),
        (["abc", "ab"], "ab", True),
        (["abc", "ab"], "a", False),
        (["abc", "ab"], "abcd", False),
        (["hello", "world"], "hello", True),
        (["hello", "world"], "hell", False),
        (["hello", "world"], "world", True),
    ],
)
def test_insert_and_search(
    words: Collection[str], search_word: str, expected: bool
) -> None:
    t = Trie.of(words)
    assert t.search(search_word) == expected


@pytest.mark.parametrize(
    "words,prefix,expected",
    [
        (["abc", "abc", "ab", "a"], "abc", True),
        (["abc", "abc", "ab", "a"], "ab", True),
        (["abc", "abc", "ab", "a"], "a", True),
        (["a", "ab", "abc", "a"], "a", True),
        (["a", "ab", "abc", "a"], "ab", True),
        (["a", "ab", "abc", "a"], "abc", True),
        (["abc"], "ab", True),
        (["abc"], "a", True),
        (["abc"], "", True),
        (["abc"], "def", False),
        (["hello", "world"], "hel", True),
        (["hello", "world"], "wor", True),
        (["hello", "world"], "worlds", False),
        (["cat", "catch"], "cat", True),
        (["cat", "catch"], "ca", True),
    ],
)
def test_prefix(words: Collection[str], prefix: str, expected: bool) -> None:
    t = Trie.of(words)
    assert t.prefix(prefix) == expected


def test_large_trie() -> None:
    words = [
        "gbmjkzcmqszxmqcvbfsliqw",
        "obclnifcwguputqftvmmhjz",
        "iaxnmhnercagdkkufmfevxz",
        "oygakojkfbaypyqhmmdjwjo",
        "kuucifueyjwswisheucjias",
        "ponhupdhldvkdxjgurbzwdt",
        "fptpgoojzjxdzrcbucfpsih",
        "kotnzieoualxiqtpoclibwb",
        "lkyqztajjqwdsguxyayfooe",
        "ghmcxiyccsdukcdhdlkorgh",
        "yesrfquvjcvmsujfdwxlgrp",
        "wyowzazdxgteecskyqbhntb",
        "mbpsmzzkaffrkxlfipcuhwe",
        "oqluzkrfpgzhmnxxohxuiow",
        "mncxvdtynwvaicoovsqbxbn",
        "ktoeprwcdadsqbsvieehwxe",
        "protmxssmtuxieencnhthcj",
        "dafjvthrbjarhwmslhwqvwe",
        "kezpnvydhlgzqrujsnirtza",
        "rmqunmyjubvfowhambblprn",
        "izjnywvlmtbfmymxwvxxvqe",
        "nilnyxjdfrshqahuebazlcc",
        "tiaaoxxnjmpgiucgiaeikgy",
        "ssbqrxivaidjewtvzcyqlwf",
        "vfpfwhofwwrjwqsavjsecun",
        "jlbqfkvqztzzdpjpumsubsn",
    ]
    t = Trie.of(words)
    assert t.search("gbmjkzcmqszxmqcvbfsliqw")
    assert t.search("obclnifcwguputqftvmmhjz")
    assert t.search("iaxnmhnercagdkkufmfevxz")
    assert t.search("oygakojkfbaypyqhmmdjwjo")
    assert t.search("kuucifueyjwswisheucjias")
    assert t.search("ponhupdhldvkdxjgurbzwdt")
    assert t.search("fptpgoojzjxdzrcbucfpsih")
    assert t.search("kotnzieoualxiqtpoclibwb")
    assert t.search("lkyqztajjqwdsguxyayfooe")
    assert t.search("ghmcxiyccsdukcdhdlkorgh")
    assert t.search("yesrfquvjcvmsujfdwxlgrp")
    assert t.search("wyowzazdxgteecskyqbhntb")
    assert t.search("mbpsmzzkaffrkxlfipcuhwe")
    assert t.search("oqluzkrfpgzhmnxxohxuiow")
    assert t.search("mncxvdtynwvaicoovsqbxbn")
    assert t.search("ktoeprwcdadsqbsvieehwxe")
    assert t.search("protmxssmtuxieencnhthcj")
    assert t.search("dafjvthrbjarhwmslhwqvwe")
    assert t.search("kezpnvydhlgzqrujsnirtza")
    assert t.search("rmqunmyjubvfowhambblprn")
    assert t.search("izjnywvlmtbfmymxwvxxvqe")
    assert t.search("nilnyxjdfrshqahuebazlcc")
    assert t.search("tiaaoxxnjmpgiucgiaeikgy")
    assert t.search("ssbqrxivaidjewtvzcyqlwf")
    assert t.search("vfpfwhofwwrjwqsavjsecun")
    assert t.search("jlbqfkvqztzzdpjpumsubsn")


if __name__ == "__main__":
    pytest.main([__file__])
