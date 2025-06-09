import array
import hashlib
from collections.abc import Collection
from dataclasses import dataclass
from typing import assert_never

import pytest


@dataclass(frozen=True, kw_only=True)
class Evicted:
    bucket_index: int
    fingerprint: int


@dataclass(frozen=True, kw_only=True)
class MaxSwapsExceeded:
    evicted: Evicted


class CuckooFilter:
    def __init__(
        self,
        num_buckets: int,
        max_bucket_size: int = 8,
        max_swaps: int = 100,
    ):
        self._num_buckets = num_buckets
        self._max_bucket_size = max_bucket_size
        self._buckets = array.array("I", [0] * (num_buckets * max_bucket_size))
        self._max_swaps = max_swaps

    def _fingerprint(self, item: str) -> int:
        return (
            fp
            if (
                fp := int.from_bytes(
                    hashlib.sha256(item.encode()).digest()[:4], byteorder="big"
                )
            )
            != 0
            else 1
        )

    def _primary_bucket_idx(self, item: str) -> int:
        return int(hashlib.sha256(item.encode()).hexdigest(), 16) % self._num_buckets

    def _alternative_bucket_idx(self, fingerprint: int, primary_bucket: int) -> int:
        return primary_bucket ^ (fingerprint % self._num_buckets)

    def _bucket_start_idx(self, bucket_index: int) -> int:
        return bucket_index * self._max_bucket_size

    def _get_bucket_fingerprints(self, bucket_index: int) -> Collection[int]:
        start = self._bucket_start_idx(bucket_index)
        return [
            fp for fp in self._buckets[start : start + self._max_bucket_size] if fp != 0
        ]

    def _add_to_bucket(self, bucket_index: int, fingerprint: int) -> None | Evicted:
        start = self._bucket_start_idx(bucket_index)
        for i in range(self._max_bucket_size):
            if self._buckets[start + i] == 0:
                self._buckets[start + i] = fingerprint
                return None
        evicted = Evicted(bucket_index=bucket_index, fingerprint=self._buckets[start])
        self._buckets[start] = fingerprint
        return evicted

    def add(self, item: str) -> None | Evicted | MaxSwapsExceeded:
        fingerprint = self._fingerprint(item)
        primary_bucket = self._primary_bucket_idx(item)

        if not self._bucket_full(primary_bucket):
            return self._add_to_bucket(primary_bucket, fingerprint)
        else:
            return self._swap(fingerprint, primary_bucket)

    def _bucket_full(self, bucket_index: int) -> bool:
        return all(
            self._buckets[self._bucket_start_idx(bucket_index) + i] != 0
            for i in range(self._max_bucket_size)
        )

    def _swap(self, fingerprint: int, bucket_index: int) -> None | MaxSwapsExceeded:
        current_fingerprint = fingerprint
        current_bucket = bucket_index

        for _ in range(self._max_swaps):
            match self._add_to_bucket(current_bucket, current_fingerprint):
                case None:
                    return None
                case Evicted() as evicted:
                    current_fingerprint = evicted.fingerprint
                    current_bucket = self._alternative_bucket_idx(
                        evicted.fingerprint, evicted.bucket_index
                    )
                case never:
                    assert_never(never)

        return MaxSwapsExceeded(
            evicted=Evicted(
                bucket_index=current_bucket, fingerprint=current_fingerprint
            )
        )

    def contains(self, item: str) -> bool:
        fingerprint = self._fingerprint(item)
        primary_bucket = self._primary_bucket_idx(item)
        return fingerprint in self._get_bucket_fingerprints(
            primary_bucket
        ) or fingerprint in self._get_bucket_fingerprints(
            self._alternative_bucket_idx(fingerprint, primary_bucket)
        )

    def delete(self, item: str) -> bool:
        fingerprint = self._fingerprint(item)
        primary_bucket = self._primary_bucket_idx(item)

        return self._delete_from_bucket(
            fingerprint, primary_bucket
        ) or self._delete_from_bucket(
            fingerprint, self._alternative_bucket_idx(fingerprint, primary_bucket)
        )

    def _delete_from_bucket(self, fingerprint: int, bucket_index: int) -> bool:
        start = self._bucket_start_idx(bucket_index)
        for i in range(self._max_bucket_size):
            if self._buckets[start + i] == fingerprint:
                self._buckets[start + i] = 0
                return True
        return False

    def __contains__(self, item: str) -> bool:
        return self.contains(item)


def test_insert():
    cf = CuckooFilter(num_buckets=100)
    cf.add("hello")
    assert "hello" in cf
    assert "world" not in cf


def test_delete():
    cf = CuckooFilter(num_buckets=100)
    cf.add("hello")
    assert "hello" in cf
    cf.delete("hello")
    assert "hello" not in cf


def test_delete_idempotent():
    cf = CuckooFilter(num_buckets=100)
    cf.add("hello")
    assert "hello" in cf
    cf.delete("hello")
    cf.delete("hello")
    assert "hello" not in cf


def test_max_swaps():
    cf = CuckooFilter(num_buckets=1, max_swaps=0, max_bucket_size=1)
    cf.add("hello")
    assert "hello" in cf
    assert isinstance(cf.add("world"), MaxSwapsExceeded)


if __name__ == "__main__":
    pytest.main([__file__])
