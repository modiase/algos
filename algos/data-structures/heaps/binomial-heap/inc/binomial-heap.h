#ifndef BINOMIAL_HEAP_H
#define BINOMIAL_HEAP_H

#include <stdbool.h>

/*
 * BINOMIAL HEAP PUBLIC INTERFACE
 *
 * A binomial heap is a priority queue data structure that supports:
 * - Insert, find-min, extract-min in O(log n) time
 * - Union operation in O(log n) time (much faster than binary heaps)
 * - Decrease-key and delete operations in O(log n) time
 *
 * The heap consists of a collection of binomial trees that satisfy:
 * 1. Each tree follows the min-heap property
 * 2. At most one tree of each degree (2^k nodes)
 * 3. Trees are stored in increasing order of degree
 */

/* Forward declarations - users don't need to know internal structure */
struct binomial_node;
struct binomial_heap;

/* ========== HEAP LIFECYCLE ========== */

/*
 * Create an empty binomial heap
 * Returns: pointer to new heap, or NULL on allocation failure
 * Time complexity: O(1)
 */
struct binomial_heap *binomial_heap_create(void);

/*
 * Destroy the heap and free all memory
 * Parameters: heap - the heap to destroy (can be NULL)
 * Time complexity: O(n)
 */
void binomial_heap_destroy(struct binomial_heap *heap);

/* ========== CORE OPERATIONS ========== */

/*
 * Insert a new key into the heap
 * Parameters: heap - target heap, key - value to insert
 * Returns: true on success, false on failure
 * Time complexity: O(log n)
 */
bool binomial_heap_insert(struct binomial_heap *heap, int key);

/*
 * Find the minimum key in the heap
 * Parameters: heap - the heap to search
 * Returns: minimum key, or INT_MAX if heap is empty
 * Time complexity: O(log n)
 */
int binomial_heap_minimum(struct binomial_heap *heap);

/*
 * Extract and return the minimum key from the heap
 * Parameters: heap - the heap to extract from
 * Returns: minimum key, or INT_MAX if heap is empty
 * Time complexity: O(log n)
 */
int binomial_heap_extract_min(struct binomial_heap *heap);

/*
 * Union two heaps into a single heap
 * Parameters: heap1, heap2 - heaps to merge
 * Returns: pointer to merged heap, or NULL on failure
 * Note: Original heaps should not be used after union
 * Time complexity: O(log n)
 */
struct binomial_heap *binomial_heap_union(struct binomial_heap *heap1, struct binomial_heap *heap2);

/* ========== UTILITY OPERATIONS ========== */

/*
 * Check if the heap is empty
 * Parameters: heap - the heap to check
 * Returns: true if empty or NULL, false otherwise
 * Time complexity: O(1)
 */
bool binomial_heap_is_empty(struct binomial_heap *heap);

/*
 * Get the number of elements in the heap
 * Parameters: heap - the heap to measure
 * Returns: number of elements, or 0 if heap is NULL
 * Time complexity: O(1)
 */
int binomial_heap_size(struct binomial_heap *heap);

/* ========== ADVANCED OPERATIONS ========== */

/*
 * Decrease the key of a specific node
 * Parameters: heap - the heap containing the node
 *            node - pointer to the node to modify
 *            new_key - the new (smaller) key value
 * Note: new_key must be smaller than current key
 * Time complexity: O(log n)
 */
void binomial_heap_decrease_key(struct binomial_heap *heap, struct binomial_node *node, int new_key);

/*
 * Delete a specific node from the heap
 * Parameters: heap - the heap containing the node
 *            node - pointer to the node to delete
 * Time complexity: O(log n)
 */
void binomial_heap_delete(struct binomial_heap *heap, struct binomial_node *node);

#endif /* BINOMIAL_HEAP_H */
