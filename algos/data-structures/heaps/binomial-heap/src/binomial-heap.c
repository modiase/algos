#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

#include "binomial-heap.h"

struct binomial_node {
    int key;                       /* The value stored in this node */
    int degree;                    /* Number of children (also the order of the tree if this node is the root) */
    struct binomial_node *parent;  /* Pointer to the parent node */
    struct binomial_node *child;   /* Pointer to the leftmost child */
    struct binomial_node *sibling; /* Pointer to the next sibling in the same level */
};

struct binomial_heap {
    struct binomial_node *head; /* Pointer to the root of the leftmost tree */
    int size;                   /* Number of elements in the heap */
};

/*
 * Create a new binomial node with given key
 * Time complexity: O(1)
 */
struct binomial_node *binomial_node_create(int key)
{
    struct binomial_node *node = malloc(sizeof(struct binomial_node));
    if (!node)
        return NULL;

    node->key = key;
    node->degree = 0;
    node->parent = NULL;
    node->child = NULL;
    node->sibling = NULL;

    return node;
}

/*
 * Recursively destroy a binomial tree rooted at node
 * Time complexity: O(n) where n is number of nodes in the tree
 */
void binomial_node_destroy(struct binomial_node *node)
{
    if (!node)
        return;

    struct binomial_node *child = node->child;
    while (child) {
        struct binomial_node *next_sibling = child->sibling;
        binomial_node_destroy(child);
        child = next_sibling;
    }

    free(node);
}

/* ========== TREE OPERATIONS ========== */

/*
 * Link two binomial trees of the same degree
 * Makes the tree with larger root the child of the one with smaller root
 * This maintains the min-heap property
 * Time complexity: O(1)
 */
struct binomial_node *binomial_tree_link(struct binomial_node *tree1, struct binomial_node *tree2)
{
    if (!tree1)
        return tree2;
    if (!tree2)
        return tree1;

    if (tree1->key > tree2->key) {
        struct binomial_node *temp = tree1;
        tree1 = tree2;
        tree2 = temp;
    }

    tree2->parent = tree1;
    tree2->sibling = tree1->child;
    tree1->child = tree2;
    tree1->degree++;

    return tree1;
}

/*
 * Merge two binomial heaps' root lists into a single sorted list
 * The result is sorted by degree (increasing order)
 * Time complexity: O(log n)
 */
struct binomial_node *binomial_tree_merge(struct binomial_node *head1, struct binomial_node *head2)
{
    if (!head1)
        return head2;
    if (!head2)
        return head1;

    struct binomial_node *merged_head = NULL;
    struct binomial_node *current = NULL;

    if (head1->degree <= head2->degree) {
        merged_head = current = head1;
        head1 = head1->sibling;
    } else {
        merged_head = current = head2;
        head2 = head2->sibling;
    }

    while (head1 && head2) {
        if (head1->degree <= head2->degree) {
            current->sibling = head1;
            head1 = head1->sibling;
        } else {
            current->sibling = head2;
            head2 = head2->sibling;
        }
        current = current->sibling;
    }

    current->sibling = head1 ? head1 : head2;

    return merged_head;
}

/* ========== HEAP OPERATIONS ========== */

/*
 * Create an empty binomial heap
 * Time complexity: O(1)
 */
struct binomial_heap *binomial_heap_create(void)
{
    struct binomial_heap *heap = malloc(sizeof(struct binomial_heap));
    if (!heap)
        return NULL;

    heap->head = NULL;
    heap->size = 0;

    return heap;
}

/*
 * Destroy the entire binomial heap and free all memory
 * Time complexity: O(n)
 */
void binomial_heap_destroy(struct binomial_heap *heap)
{
    if (!heap)
        return;

    /* Destroy all trees in the heap */
    struct binomial_node *current = heap->head;
    while (current) {
        struct binomial_node *next = current->sibling;
        binomial_node_destroy(current);
        current = next;
    }

    free(heap);
}

/*
 * Check if the heap is empty
 * Time complexity: O(1)
 */
bool binomial_heap_is_empty(struct binomial_heap *heap)
{
    return heap == NULL || heap->head == NULL;
}

/*
 * Get the number of elements in the heap
 * Time complexity: O(1)
 */
int binomial_heap_size(struct binomial_heap *heap)
{
    return heap ? heap->size : 0;
}

/*
 * Insert a new key into the binomial heap
 * Creates a new heap with single element and unions it with existing heap
 * Time complexity: O(log n)
 */
bool binomial_heap_insert(struct binomial_heap *heap, int key)
{
    if (!heap)
        return false;

    struct binomial_node *new_node = binomial_node_create(key);
    if (!new_node)
        return false;

    struct binomial_heap *temp_heap = binomial_heap_create();
    if (!temp_heap) {
        free(new_node);
        return false;
    }

    temp_heap->head = new_node;
    temp_heap->size = 1;

    struct binomial_heap *result = binomial_heap_union(heap, temp_heap);
    if (!result) {
        binomial_heap_destroy(temp_heap);
        return false;
    }

    /* Update the original heap */
    heap->head = result->head;
    heap->size = result->size;

    free(result);
    free(temp_heap);

    return true;
}

/*
 * Find the minimum key in the heap
 * Scans through all root nodes to find the minimum
 * Time complexity: O(log n)
 *
 * Note: we don't use a min pointer because the
 * find-min operation is never used outside the context of delete-min which is
 * O(log n) anyway and so overall gain is made for the additional implementation
 * complexity.
 */
int binomial_heap_minimum(struct binomial_heap *heap)
{
    if (binomial_heap_is_empty(heap)) {
        return INT_MAX; /* Error value - heap is empty */
    }

    int min_key = heap->head->key;
    struct binomial_node *current = heap->head->sibling;

    while (current) {
        if (current->key < min_key) {
            min_key = current->key;
        }
        current = current->sibling;
    }

    return min_key;
}

/*
 * Extract and return the minimum key from the heap
 * 1. Find the minimum root
 * 2. Remove it from the root list
 * 3. Reverse the order of its children to create a new heap
 * 4. Union the remaining trees with the new heap
 * Time complexity: O(log n)
 */
int binomial_heap_extract_min(struct binomial_heap *heap)
{
    if (binomial_heap_is_empty(heap)) {
        return INT_MAX; /* Error value */
    }

    /* Find the minimum root and its predecessor */
    struct binomial_node *min_node = heap->head;
    struct binomial_node *min_prev = NULL;
    struct binomial_node *current = heap->head;
    struct binomial_node *prev = NULL;

    while (current) {
        if (current->key < min_node->key) {
            min_node = current;
            min_prev = prev;
        }
        prev = current;
        current = current->sibling;
    }

    int min_key = min_node->key;

    /* Remove min_node from the root list */
    if (min_prev) {
        min_prev->sibling = min_node->sibling;
    } else {
        heap->head = min_node->sibling;
    }

    /* Reverse the children of min_node to create a new heap */
    struct binomial_node *new_head = NULL;
    struct binomial_node *child = min_node->child;

    while (child) {
        struct binomial_node *next = child->sibling;
        child->sibling = new_head;
        child->parent = NULL;
        new_head = child;
        child = next;
    }

    struct binomial_heap *child_heap = binomial_heap_create();
    if (child_heap) {
        child_heap->head = new_head;

        struct binomial_heap *result = binomial_heap_union(heap, child_heap);
        if (result) {
            heap->head = result->head;
            heap->size = result->size - 1; /* Subtract 1 for the extracted node */
            free(result);
        }

        free(child_heap);
    }

    free(min_node);

    return min_key;
}

/*
 * Union two binomial heaps into a single heap
 * 1. Merge the root lists of both heaps
 * 2. Link trees of the same degree
 * Time complexity: O(log n)
 */
struct binomial_heap *binomial_heap_union(struct binomial_heap *heap1, struct binomial_heap *heap2)
{
    if (!heap1 && !heap2)
        return NULL;
    if (!heap1)
        return heap2;
    if (!heap2)
        return heap1;

    struct binomial_heap *result = binomial_heap_create();
    if (!result)
        return NULL;

    /* Merge the root lists */
    result->head = binomial_tree_merge(heap1->head, heap2->head);
    result->size = heap1->size + heap2->size;

    if (!result->head)
        return result;

    /* Link trees of same degree */
    struct binomial_node *prev = NULL;
    struct binomial_node *current = result->head;
    struct binomial_node *next = current->sibling;

    while (next) {
        /* Case 1: current and next have different degrees, or
         * there are three consecutive trees with same degree */
        if (current->degree != next->degree ||
            (next->sibling && next->sibling->degree == current->degree)) {
            prev = current;
            current = next;
        }
        /* Case 2: current and next have same degree - link them */
        else {
            if (current->key <= next->key) {
                /* Link next as child of current */
                current->sibling = next->sibling;
                binomial_tree_link(current, next);
            } else {
                /* Link current as child of next */
                if (prev) {
                    prev->sibling = next;
                } else {
                    result->head = next;
                }
                binomial_tree_link(next, current);
                current = next;
            }
        }
        next = current->sibling;
    }

    return result;
}

/*
 * Decrease the key of a given node
 * Bubbles up the node to maintain heap property
 * Time complexity: O(log n)
 */
void binomial_heap_decrease_key(struct binomial_heap *heap, struct binomial_node *node, int new_key)
{
    if (!heap || !node || new_key > node->key)
        return;

    node->key = new_key;

    /* Bubble up to maintain min-heap property */
    struct binomial_node *current = node;
    struct binomial_node *parent = current->parent;

    while (parent && current->key < parent->key) {
        /* Swap keys */
        int temp = current->key;
        current->key = parent->key;
        parent->key = temp;

        current = parent;
        parent = current->parent;
    }
}

/*
 * Delete a node from the heap
 * 1. Decrease its key to negative infinity
 * 2. Extract the minimum (which will be this node)
 * Time complexity: O(log n)
 */
void binomial_heap_delete(struct binomial_heap *heap, struct binomial_node *node)
{
    if (!heap || !node)
        return;

    /* Decrease key to minimum possible value */
    binomial_heap_decrease_key(heap, node, INT_MIN);

    /* Extract minimum (which is now this node) */
    binomial_heap_extract_min(heap);
}
