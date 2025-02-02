from collections import defaultdict


class Trie:
    SENTINEL = -1

    class Node:
        def __init__(self):
            self._edges = defaultdict(Trie.Node)

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


def test_insert_and_search():
    t = Trie()
    t.insert("abc")
    assert t.search("abc")
    assert not t.search("def")
    assert not t.search("ab")
    assert not t.search("a")
    assert not t.search("abcd")

    t.insert("ab")
    assert t.search("abc")
    assert t.search("ab")
    assert not t.search("a")
    assert not t.search("abcd")


def test_prefix():
    t = Trie()
    t.insert("abc")
    assert t.prefix("abc")
    assert t.prefix("ab")
    assert t.prefix("a")
    assert t.prefix("")
    assert not t.prefix("def")


def tests():
    test_insert_and_search()

    test_prefix()

    print("All tests successful")


if __name__ == "__main__":
    tests()
