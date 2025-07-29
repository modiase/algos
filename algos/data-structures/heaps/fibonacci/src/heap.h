#ifndef FIBONACCI_HEAP
#define FIBONACCI_HEAP

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#include "utils.h"

/**
 * Fibonacci heap struct
 */
struct fibonacci_heap;

/**
 * Create a new fibonacci heap
 */
struct fibonacci_heap *fib_heap_create();

/**
 * Destroy a fibonacci heap
 */
void fib_heap_destroy(struct fibonacci_heap *heap);

/**
 * Insert a new node into the fibonacci heap
 * @heap: the fibonacci heap
 * @key: the key value for the new node
 * @data: the data associated with the key
 */
void fib_heap_insert(struct fibonacci_heap *heap, int key, void *data);

#endif