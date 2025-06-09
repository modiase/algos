import hashlib
from array import array
from collections.abc import Collection

import pytest


class BloomFilter:
    def __init__(self, size: int):
        self._size = size
        self._bit_array = array("B", [0] * size)
        self._hash_functions = [
            hashlib.sha256,
            hashlib.sha512,
            hashlib.sha3_256,
            hashlib.sha3_512,
        ]

    def _hash(self, item: str) -> Collection[int]:
        return tuple(
            int(hash_function(item.encode()).hexdigest(), 16) % self._size
            for hash_function in self._hash_functions
        )

    def add(self, item: str) -> None:
        for value in self._hash(item):
            self._bit_array[value] = 1

    def contains(self, item: str) -> bool:
        return all(self._bit_array[value] for value in self._hash(item))

    def __contains__(self, item: str) -> bool:
        return self.contains(item)

    def __setitem__(self, key: str, value: bool) -> None:
        if value:
            return self.add(key)
        raise NotImplementedError("BloomFilter does not support item deletion")

    def __getitem__(self, key: str) -> bool:
        return self.contains(key)


def test_contains():
    bloom_filter = BloomFilter(100)
    bloom_filter["hello"] = True
    assert "hello" in bloom_filter
    assert "world" not in bloom_filter


def test_does_not_contain():
    bloom_filter = BloomFilter(100)
    bloom_filter["hello"] = True
    assert "world" not in bloom_filter


def test_deletion_throws():
    bloom_filter = BloomFilter(100)
    with pytest.raises(NotImplementedError):
        bloom_filter["hello"] = False


if __name__ == "__main__":
    pytest.main([__file__])
