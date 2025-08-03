#ifndef DISJOINT_SET_H
#define DISJOINT_SET_H

#include <stdio.h>
#include <stdlib.h>

/**
 * @brief Defines the structure for a node in the disjoint set.
 *
 * Each node can be a data element or a special head (representative) node.
 * The list is circular and doubly-linked. The head node's `key` and `size`
 * are used to store metadata about the set (size), while data nodes use `key`
 * for their value.
 */
typedef struct Node {
    int key;		// The data value for an element node.
    int size;		// The size of the set (only used by the head node).
    struct Node *next;	// Pointer to the next node in the circular list.
    struct Node *prev;	// Pointer to the previous node in the circular list.
    struct Node *representative;	// Pointer to the head node of the set.
} Node;

/**
 * @brief Creates a new set containing a single element, x.
 *
 * This function allocates two nodes: one head node to act as the set's
 * representative and one data node for the element `x`. It initializes a
 * circular, doubly-linked list containing these two nodes.
 *
 * @param x The integer value to store in the new set.
 * @return A pointer to the data node containing x. Returns NULL on failure.
 */
static inline Node *make_set(int x)
{
    Node *head_node = (Node *) malloc(sizeof(Node));
    Node *data_node = (Node *) malloc(sizeof(Node));

    if (head_node == NULL || data_node == NULL) {
        fprintf(stderr,
                "Error: Memory allocation failed in make_set.\n");
        free(head_node);
        free(data_node);
        return NULL;
    }

    head_node->size = 1;
    head_node->representative = head_node;
    head_node->key = -1;

    data_node->key = x;
    data_node->representative = head_node;
    data_node->size = 0;

    head_node->next = data_node;
    head_node->prev = data_node;
    data_node->next = head_node;
    data_node->prev = head_node;

    return data_node;
}

/**
 * @brief Finds the representative of the set containing node x.
 *
 * Since every node stores a direct pointer to its representative, this is an O(1)
 * operation.
 *
 * @param x A pointer to a node in the set.
 * @return A pointer to the representative (head) node of the set.
 */
static inline Node *find_set(Node *x)
{
    if (x == NULL) {
        return NULL;
    }
    return x->representative;
}

/**
 * @brief Merges the two sets containing nodes a and b.
 *
 * This function implements the union-by-size heuristic. The smaller set is
 * always merged into the larger set to maintain efficiency. All nodes from the
 * smaller set are updated to point to the larger set's representative.
 *
 * @param a A pointer to a node in the first set.
 * @param b A pointer to a node in the second set.
 * @return A pointer to the representative of the new, merged set.
 */
static inline Node *union_sets(Node *a, Node *b)
{
    Node *repA = find_set(a);
    Node *repB = find_set(b);

    if (repA == repB) {
        return repA;
    }

    if (repA->size < repB->size) {
        Node *temp = repA;
        repA = repB;
        repB = temp;
    }

    Node *current = repB->next;
    while (current != repB) {
        current->representative = repA;
        current = current->next;
    }

    Node *tailA = repA->prev;
    Node *headB = repB->next;
    Node *tailB = repB->prev;

    tailA->next = headB;
    headB->prev = tailA;
    tailB->next = repA;
    repA->prev = tailB;

    repA->size += repB->size;

    free(repB);

    return repA;
}

/**
 * @brief Frees all memory associated with a set given its representative.
 *
 * This function deallocates all nodes in the set, including both data nodes
 * and the representative (head) node. It traverses the circular linked list
 * starting from the representative's next node and frees each data node,
 * then finally frees the representative node itself.
 *
 * @param representative The head node of the set to destroy. If NULL, no action is taken.
 */
static inline void destroy_set(Node *representative)
{
    if (representative == NULL)
        return;

    Node *current = representative->next;
    while (current != representative) {
        Node *to_free = current;
        current = current->next;
        free(to_free);
    }
    free(representative);
}

#endif				// DISJOINT_SET_H
