import hashlib
from collections.abc import Collection

import pytest


class CountingBloomFilter:
    def __init__(self, size: int):
        self.size = size
        self.bit_array = [0] * size
        self.hash_functions = [
            hashlib.sha256,
            hashlib.sha512,
            hashlib.sha3_256,
            hashlib.sha3_512,
        ]

    def _hash(self, item: str) -> Collection[int]:
        return tuple(
            int(hash_function(item.encode()).hexdigest(), 16) % self.size
            for hash_function in self.hash_functions
        )

    def add(self, item: str) -> None:
        for value in self._hash(item):
            self.bit_array[value] += 1

    def contains(self, item: str) -> bool:
        return all(self.bit_array[value] > 0 for value in self._hash(item))

    def __contains__(self, item: str) -> bool:
        return self.contains(item)

    def __setitem__(self, key: str, value: bool) -> None:
        return self.add(key) if value else self.remove(key)

    def __getitem__(self, key: str) -> bool:
        return self.contains(key)

    def remove(self, item: str) -> None:
        if item not in self:
            return None
        for value in self._hash(item):
            self.bit_array[value] -= 1


def test_contains():
    bloom_filter = CountingBloomFilter(100)
    bloom_filter["hello"] = True
    assert "hello" in bloom_filter
    assert "world" not in bloom_filter


def test_does_not_contain():
    bloom_filter = CountingBloomFilter(100)
    bloom_filter["hello"] = True
    assert "world" not in bloom_filter


def test_deletion():
    bloom_filter = CountingBloomFilter(100)
    bloom_filter["hello"] = True
    assert "hello" in bloom_filter
    bloom_filter["hello"] = False
    assert "hello" not in bloom_filter


def test_delete_idempotent():
    bloom_filter = CountingBloomFilter(100)
    bloom_filter["hello"] = True
    bloom_filter["hello"] = False
    bloom_filter["hello"] = False
    assert "hello" not in bloom_filter


if __name__ == "__main__":
    pytest.main([__file__])
