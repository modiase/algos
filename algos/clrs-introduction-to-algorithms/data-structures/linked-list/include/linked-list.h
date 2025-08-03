#include <stddef.h>

#ifndef __LINKED_LIST_H
#define __LINKED_LIST_H
struct Node {
    void *data;
    struct Node *next;
};
const static struct Node NIL = (struct Node)
{
    .data = NULL,.next = NULL
};

struct Node *LinkedList__prepend(const struct Node *head, struct Node *item);
#endif
