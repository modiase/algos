import gc
import weakref

import pytest
import ring_buffer


@pytest.fixture
def buffer():
    """Create a fresh ring buffer for tests."""
    return ring_buffer.RingBuffer()


@pytest.fixture
def small_buffer():
    """Create a ring buffer for resize testing."""
    return ring_buffer.RingBuffer()


def test_create_buffer():
    """Test creating ring buffers."""
    rb = ring_buffer.RingBuffer()
    assert rb.size() == 0
    assert rb.is_empty()
    assert len(rb) == 0
    assert not bool(rb)


def test_single_enqueue_dequeue(buffer):
    """Test basic enqueue and dequeue operations."""
    buffer.enqueue("hello")
    assert buffer.size() == 1
    assert not buffer.is_empty()
    assert bool(buffer)

    result = buffer.dequeue()
    assert result == "hello"
    assert buffer.size() == 0
    assert buffer.is_empty()


@pytest.mark.parametrize(
    "items",
    [
        ["single"],
        ["first", "second"],
        ["a", "b", "c", "d", "e"],
        [f"item_{i}" for i in range(10)],
        [f"item_{i}" for i in range(100)],
    ],
)
def test_fifo_ordering(buffer, items):
    """Test first-in-first-out ordering with different sized lists."""
    for item in items:
        buffer.enqueue(item)

    results = []
    while not buffer.is_empty():
        results.append(buffer.dequeue())

    assert results == items


@pytest.mark.parametrize(
    "value",
    [
        42,
        3.14,
        "string",
        [1, 2, 3],
        {"key": "value"},
        (1, 2, 3),
        set([1, 2, 3]),
        frozenset([4, 5, 6]),
        None,
        True,
        False,
        b"bytes",
        complex(1, 2),
    ],
)
def test_different_types(buffer, value):
    """Test storing different Python types."""
    buffer.enqueue(value)
    result = buffer.dequeue()
    assert result == value
    assert type(result) == type(value)


def test_callable_objects(buffer):
    """Test storing and retrieving callable objects."""
    test_func = lambda x: x * 2
    buffer.enqueue(test_func)
    result = buffer.dequeue()
    assert callable(result)
    assert result(5) == 10


def test_empty_dequeue_raises_error(buffer):
    """Test that dequeuing from empty buffer raises IndexError."""
    with pytest.raises(IndexError):
        buffer.dequeue()


@pytest.mark.parametrize("num_items", [3, 10, 50, 100])
def test_auto_resize_growth(small_buffer, num_items):
    """Test automatic growth with different numbers of items."""
    items = [f"item_{i}" for i in range(num_items)]
    for item in items:
        small_buffer.enqueue(item)

    assert small_buffer.size() == num_items

    results = []
    while not small_buffer.is_empty():
        results.append(small_buffer.dequeue())

    assert results == items


@pytest.mark.parametrize(
    "operations",
    [
        # (enqueue_count, dequeue_count)
        (5, 2),
        (10, 5),
        (20, 15),
        (100, 90),
    ],
)
def test_mixed_operations(buffer, operations):
    """Test mixing enqueue and dequeue operations."""
    enqueue_count, dequeue_count = operations

    # Enqueue items
    items = [f"item_{i}" for i in range(enqueue_count)]
    for item in items:
        buffer.enqueue(item)

    assert buffer.size() == enqueue_count

    # Dequeue some items
    results = []
    for _ in range(dequeue_count):
        results.append(buffer.dequeue())

    # Should get first dequeue_count items in order
    expected = items[:dequeue_count]
    assert results == expected
    assert buffer.size() == enqueue_count - dequeue_count


def test_clear_functionality(buffer):
    """Test clearing the buffer."""
    # Add items beyond initial capacity
    for i in range(10):
        buffer.enqueue(f"item_{i}")

    assert buffer.size() > 0

    buffer.clear()

    assert buffer.size() == 0
    assert buffer.is_empty()
    assert len(buffer) == 0
    assert not bool(buffer)


@pytest.mark.parametrize(
    "pattern",
    [
        # (add, remove, add, remove, ...)
        [5, 2, 3, 1, 10, 8],
        [1, 1, 1, 1, 1, 1],
        [20, 10, 15, 5, 30, 25],
    ],
)
def test_alternating_operations(buffer, pattern):
    """Test alternating enqueue/dequeue patterns."""
    all_items = []
    current_id = 0

    for i, count in enumerate(pattern):
        if i % 2 == 0:  # Even indices = enqueue
            items = [f"item_{current_id + j}" for j in range(count)]
            all_items.extend(items)
            current_id += count

            for item in items:
                buffer.enqueue(item)
        else:  # Odd indices = dequeue
            removed = []
            for _ in range(min(count, buffer.size())):
                removed.append(buffer.dequeue())

            # Should match the expected items in order
            expected = all_items[: len(removed)]
            all_items = all_items[len(removed) :]
            assert removed == expected


def test_memory_management(buffer):
    """Test proper garbage collection of stored objects."""

    class TestObject:
        def __init__(self, value):
            self.value = value

    obj = TestObject("test")
    weak_ref = weakref.ref(obj)

    # Add to buffer and remove original reference
    buffer.enqueue(obj)
    del obj
    assert weak_ref() is not None  # Buffer should keep it alive

    # Remove from buffer
    retrieved = buffer.dequeue()
    assert retrieved.value == "test"

    # Object should now be collectible
    del retrieved
    gc.collect()
    assert weak_ref() is None


@pytest.mark.parametrize(
    "size_sequence",
    [
        [1, 5, 2, 10, 1],
        [10, 1, 20, 5, 50],
        [100, 10, 200, 50, 300],
    ],
)
def test_size_tracking(buffer, size_sequence):
    """Test size tracking during various operations."""
    for target_size in size_sequence:
        current_size = buffer.size()

        if target_size > current_size:
            # Add items
            for i in range(target_size - current_size):
                buffer.enqueue(f"fill_{i}")
        else:
            # Remove items
            for _ in range(current_size - target_size):
                if not buffer.is_empty():
                    buffer.dequeue()

        assert buffer.size() == target_size or buffer.is_empty()


@pytest.mark.parametrize(
    "batch_size,num_batches",
    [
        (10, 5),
        (50, 10),
        (100, 5),
    ],
)
def test_large_datasets(buffer, batch_size, num_batches):
    """Test handling larger amounts of data."""
    all_items = []

    # Add items in batches
    for batch in range(num_batches):
        batch_items = [f"batch_{batch}_item_{i}" for i in range(batch_size)]
        all_items.extend(batch_items)

        for item in batch_items:
            buffer.enqueue(item)

    assert buffer.size() == batch_size * num_batches

    # Remove all items and verify order
    results = []
    while not buffer.is_empty():
        results.append(buffer.dequeue())

    assert results == all_items


def test_builtin_functions(buffer):
    """Test Python builtin function integration."""
    # Test len()
    assert len(buffer) == 0

    buffer.enqueue("test1")
    assert len(buffer) == 1

    buffer.enqueue("test2")
    assert len(buffer) == 2

    # Test bool()
    assert bool(buffer) is True

    buffer.clear()
    assert bool(buffer) is False


@pytest.mark.parametrize(
    "complex_data",
    [
        {"nested": {"dict": {"with": ["mixed", "types", 123]}}},
        [{"list": "of"}, {"mixed": "dicts"}, [1, 2, 3]],
        (1, "tuple", {"with": "mixed"}, [4, 5, 6]),
        {"func": lambda x: x + 1, "list": [1, 2, 3]},
    ],
)
def test_complex_data_structures(buffer, complex_data):
    """Test storing complex nested data structures."""
    buffer.enqueue(complex_data)
    result = buffer.dequeue()

    if callable(complex_data.get("func")) if isinstance(complex_data, dict) else False:
        # Special handling for data with functions
        assert result["list"] == complex_data["list"]
        assert callable(result["func"])
    else:
        assert result == complex_data


def test_stress_operations(small_buffer):
    """Stress test with many rapid operations."""
    import random

    random.seed(42)  # Reproducible test

    items = []

    for _ in range(1000):
        if small_buffer.size() == 0 or random.choice([True, False]):
            # Enqueue
            item = f"stress_item_{len(items)}"
            small_buffer.enqueue(item)
            items.append(item)
        else:
            # Dequeue
            if items:
                expected = items.pop(0)
                result = small_buffer.dequeue()
                assert result == expected

    # Verify final state
    assert small_buffer.size() == len(items)

    # Clean up remaining items
    while items:
        expected = items.pop(0)
        result = small_buffer.dequeue()
        assert result == expected


def test_dynamic_resize():
    """Test that the buffer resizes dynamically by adding many items."""
    rb = ring_buffer.RingBuffer()

    # Fill well beyond default capacity to ensure resizing occurs
    items = [f"item_{i}" for i in range(200)]
    for item in items:
        rb.enqueue(item)

    # Should handle all items
    assert rb.size() == len(items)

    # Verify all items come out in order
    results = []
    while not rb.is_empty():
        results.append(rb.dequeue())

    assert results == items


def test_large_scale_operations():
    """Test large scale operations to ensure the buffer can handle significant growth."""
    rb = ring_buffer.RingBuffer()

    # Test with a large number of items
    num_items = 10000
    items = [f"large_item_{i}" for i in range(num_items)]

    # Add all items
    for item in items:
        rb.enqueue(item)

    assert rb.size() == num_items

    # Remove half
    removed_items = []
    for _ in range(num_items // 2):
        removed_items.append(rb.dequeue())

    # Should get first half in order
    assert removed_items == items[: num_items // 2]
    assert rb.size() == num_items - num_items // 2

    # Add more items
    additional_items = [f"additional_{i}" for i in range(100)]
    for item in additional_items:
        rb.enqueue(item)

    # Verify final state - should have second half + additional items
    expected_remaining = items[num_items // 2 :] + additional_items
    assert rb.size() == len(expected_remaining)

    # Clean up all remaining items and verify order
    final_results = []
    while not rb.is_empty():
        final_results.append(rb.dequeue())

    assert final_results == expected_remaining
