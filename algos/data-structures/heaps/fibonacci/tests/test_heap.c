#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "../src/heap.h"

// Test helper functions
static void test_heap_creation() {
    printf("Testing heap creation...\n");
    
    struct fibonacci_heap *heap = fib_heap_create();
    assert(heap != NULL);
    assert(heap->root == NULL);
    assert(heap->min == NULL);
    assert(heap->size == 0);
    assert(heap->maxSize == 1024); // MIN_SIZE
    
    printf("✓ Heap creation successful\n");
    
    fib_heap_destroy(heap);
    printf("✓ Heap destruction successful\n\n");
}

static void test_single_insert() {
    printf("Testing single node insertion...\n");
    
    struct fibonacci_heap *heap = fib_heap_create();
    
    // Insert a single node
    int key = 42;
    char *data = "test_data";
    fib_heap_insert(heap, key, data);
    
    assert(heap->size == 1);
    assert(heap->root != NULL);
    assert(heap->min != NULL);
    assert(heap->root == heap->min);
    assert(heap->root->key == key);
    assert(heap->root->data == data);
    assert(heap->root->degree == 0);
    assert(heap->root->mark == false);
    
    // Check circular doubly-linked list properties
    assert(heap->root->next == heap->root);
    assert(heap->root->prev == heap->root);
    
    printf("✓ Single node insertion successful\n");
    
    fib_heap_destroy(heap);
    printf("✓ Single node cleanup successful\n\n");
}

static void test_multiple_inserts() {
    printf("Testing multiple node insertions...\n");
    
    struct fibonacci_heap *heap = fib_heap_create();
    
    // Insert multiple nodes
    int keys[] = {10, 5, 15, 3, 8};
    char *data[] = {"data1", "data2", "data3", "data4", "data5"};
    int num_nodes = 5;
    
    for (int i = 0; i < num_nodes; i++) {
        fib_heap_insert(heap, keys[i], data[i]);
    }
    
    assert(heap->size == num_nodes);
    assert(heap->root != NULL);
    assert(heap->min != NULL);
    
    // Check that min points to the smallest key
    assert(heap->min->key == 3); // Smallest key
    
    // Verify all nodes are in the root list
    struct node *current = heap->root;
    int count = 0;
    do {
        count++;
        current = current->next;
    } while (current != heap->root);
    
    assert(count == num_nodes);
    
    printf("✓ Multiple node insertions successful\n");
    printf("✓ Min node correctly identified (key: %d)\n", heap->min->key);
    
    fib_heap_destroy(heap);
    printf("✓ Multiple nodes cleanup successful\n\n");
}

static void test_insert_with_duplicate_keys() {
    printf("Testing insertion with duplicate keys...\n");
    
    struct fibonacci_heap *heap = fib_heap_create();
    
    // Insert nodes with duplicate keys
    fib_heap_insert(heap, 5, "first");
    fib_heap_insert(heap, 5, "second");
    fib_heap_insert(heap, 5, "third");
    
    assert(heap->size == 3);
    assert(heap->min->key == 5);
    
    // All nodes should have the same key
    struct node *current = heap->root;
    do {
        assert(current->key == 5);
        current = current->next;
    } while (current != heap->root);
    
    printf("✓ Duplicate key insertion successful\n");
    
    fib_heap_destroy(heap);
    printf("✓ Duplicate keys cleanup successful\n\n");
}

static void test_insert_with_negative_keys() {
    printf("Testing insertion with negative keys...\n");
    
    struct fibonacci_heap *heap = fib_heap_create();
    
    // Insert nodes with negative keys
    fib_heap_insert(heap, 10, "positive");
    fib_heap_insert(heap, -5, "negative");
    fib_heap_insert(heap, 0, "zero");
    fib_heap_insert(heap, -10, "more_negative");
    
    assert(heap->size == 4);
    assert(heap->min->key == -10); // Most negative should be min
    
    printf("✓ Negative key insertion successful\n");
    printf("✓ Min correctly identified as most negative (key: %d)\n", heap->min->key);
    
    fib_heap_destroy(heap);
    printf("✓ Negative keys cleanup successful\n\n");
}

static void test_insert_with_null_data() {
    printf("Testing insertion with NULL data...\n");
    
    struct fibonacci_heap *heap = fib_heap_create();
    
    // Insert nodes with NULL data
    fib_heap_insert(heap, 1, NULL);
    fib_heap_insert(heap, 2, NULL);
    
    assert(heap->size == 2);
    assert(heap->root->data == NULL);
    assert(heap->root->next->data == NULL);
    
    printf("✓ NULL data insertion successful\n");
    
    fib_heap_destroy(heap);
    printf("✓ NULL data cleanup successful\n\n");
}

static void test_large_number_of_inserts() {
    printf("Testing large number of insertions...\n");
    
    struct fibonacci_heap *heap = fib_heap_create();
    
    // Insert many nodes
    int num_nodes = 1000;
    for (int i = 0; i < num_nodes; i++) {
        fib_heap_insert(heap, i, NULL);
    }
    
    assert(heap->size == num_nodes);
    assert(heap->min->key == 0); // Smallest key should be 0
    
    // Verify all nodes are present
    struct node *current = heap->root;
    int count = 0;
    do {
        count++;
        current = current->next;
    } while (current != heap->root);
    
    assert(count == num_nodes);
    
    printf("✓ Large number of insertions successful (%d nodes)\n", num_nodes);
    printf("✓ Min correctly identified (key: %d)\n", heap->min->key);
    
    fib_heap_destroy(heap);
    printf("✓ Large number cleanup successful\n\n");
}

int main() {
    printf("=== Fibonacci Heap Tests ===\n\n");
    
    test_heap_creation();
    test_single_insert();
    test_multiple_inserts();
    test_insert_with_duplicate_keys();
    test_insert_with_negative_keys();
    test_insert_with_null_data();
    test_large_number_of_inserts();
    
    printf("=== All tests passed! ===\n");
    return 0;
} 