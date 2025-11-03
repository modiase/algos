from collections import defaultdict

WORD_BOUNDARY = object()


class TrieNode:
    def __init__(self):
        self._keys = defaultdict(self.__class__)

    def search(self, prefix: str) -> list["TrieNode"] | None:
        c, remaining = prefix[0], prefix[1:]
        if c in self._keys:
            if not remaining:
                if WORD_BOUNDARY in self._keys[c]._keys:
                    return [self, self._keys[c]]
                return None
            result = self._keys[c].search(remaining)
            return None if result is None else [self, *result]
        return None

    def continuations(self, prefix: str) -> list[str]:
        if prefix == "":
            return list(self._keys)
        c, remaining = prefix[0], prefix[1:]
        if c in self._keys:
            if not remaining:
                return list(self._keys[c]._keys)
            return self._keys[c].continuations(remaining)
        return []

    def insert(self, word: str) -> list["TrieNode"]:
        if not word:
            node = self._keys[WORD_BOUNDARY]
            return [self, node]
        c, remaining = word[0], word[1:]
        node = self._keys[c]
        path = node.insert(remaining)
        return [self, *path]

    def remove(self, word: str) -> None:
        path = self.search(word)
        if path is None:
            return

        del path[-1]._keys[WORD_BOUNDARY]

        for i in range(len(word) - 1, -1, -1):
            if len(path[i + 1]._keys) == 0:
                del path[i]._keys[word[i]]
            else:
                break

    def __contains__(self, item):
        return item in self._keys

    def __iter__(self):
        return iter(self._keys)


class GraphNode:
    def __init__(self, val):
        self._val = val
        self._edges = []

    def add_edge(self, node: "GraphNode"):
        self._edges.append(node)


class Solution:
    def findWords(self, board: list[list[str]], words: list[str]) -> list[str]:
        trie = TrieNode()
        for word in words:
            trie.insert(word)

        starts = set(w[0] for w in words)
        start_nodes = []
        node_board = [[GraphNode(c) for c in row] for row in board]

        board_letters = set()
        N = len(board)
        M = len(board[0])
        for row_idx in range(N):
            for col_idx in range(M):
                node = node_board[row_idx][col_idx]
                board_letters.add(node._val)
                if node._val in starts:
                    start_nodes.append(node)
                if row_idx > 0:
                    node.add_edge(node_board[row_idx - 1][col_idx])
                if row_idx < N - 1:
                    node.add_edge(node_board[row_idx + 1][col_idx])
                if col_idx < M - 1:
                    node.add_edge(node_board[row_idx][col_idx + 1])
                if col_idx > 0:
                    node.add_edge(node_board[row_idx][col_idx - 1])

        start_node_letters = {n._val for n in start_nodes}
        search_set = set(words)
        for word in words:
            if word[0] not in start_node_letters or set(word) - board_letters:
                trie.remove(word)
                search_set.remove(word)
        found = set()

        def dfs(start: GraphNode):
            def _dfs(current: GraphNode, visited: list[GraphNode]):
                if not search_set - found:
                    return
                visited = [*visited, current]
                prefix = "".join([n._val for n in visited])
                continuations = trie.continuations(prefix)
                continuation_letters = {
                    w[0] for w in continuations if w != WORD_BOUNDARY
                }
                if WORD_BOUNDARY in continuations:
                    found.add(prefix)
                    trie.remove(prefix)
                for node in current._edges:
                    if node in visited:
                        continue
                    if node._val not in continuation_letters:
                        continue
                    _dfs(node, visited)

            return _dfs(start, [])

        for start in start_nodes:
            dfs(start)

        return list(found)
