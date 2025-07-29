#include "heap.h"

const float MIN_LOAD_FACTOR = 0.25;
const float MAX_LOAD_FACTOR = 0.75;
const uint32_t MIN_SIZE = 1024;

static struct node {
    struct node *prev;
    struct node *next;
    struct node *child;
    void *data;
    uint32_t degree;
    int32_t key;
    bool mark;
};

struct fibonacci_heap {
    struct node *root;
    struct node *min;
    uint32_t size;
    uint32_t maxSize;
};


struct fibonacci_heap *fib_heap_create() {
    struct fibonacci_heap *heap = malloc(sizeof(struct fibonacci_heap));
    CHECK_ALLOC_FATAL(heap);
    heap->root = NULL;
    heap->min = NULL;
    heap->size = 0;
    heap->maxSize = MIN_SIZE;
    return heap;
}

void fib_heap_destroy(struct fibonacci_heap *heap) {
    free(heap);
}

static bool _internal_fib_heap_resize(struct fibonacci_heap *heap) {
    bool resize = false;
    if (heap->size < heap->maxSize * MIN_LOAD_FACTOR && heap->maxSize > MIN_SIZE) {
        heap->maxSize = MAX(heap->maxSize / 2, MIN_SIZE);
        resize = true;
    } else if (heap->size > heap->maxSize * MAX_LOAD_FACTOR) {
        heap->maxSize *= 2;
        resize = true;
    }
    if (resize) {
        heap->root = realloc(heap->root, heap->maxSize * sizeof(struct node));
        CHECK_ALLOC_FATAL(heap->root);
    }
    return resize;
}

static struct node *fib_create_node(int key, void *data) {
    struct node *node = malloc(sizeof(struct node));
    CHECK_ALLOC_FATAL(node);
    node->key = key;
    node->degree = 0;
    node->mark = false;
    node->data = data;
    return node;
}

void fib_heap_insert(struct fibonacci_heap *heap, int key, void *data) {
    struct node *node = fib_create_node(key, data);
    
    // Initialize the node's pointers for circular doubly-linked list
    node->next = node;
    node->prev = node;
    
    if (heap->root == NULL) {
        // First node in the heap
        heap->root = node;
        heap->min = node;
    } else {
        // Insert at the beginning of the root list
        node->next = heap->root;
        node->prev = heap->root->prev;
        heap->root->prev->next = node;
        heap->root->prev = node;
        heap->root = node;
    }
    
    // Update min pointer if this node has a smaller key
    if (heap->min == NULL || node->key < heap->min->key) {
        heap->min = node;
    }
    
    heap->size++;
    _internal_fib_heap_resize(heap);
}