#include "linked-list.h"

struct Node *LinkedList__prepend(const struct Node *head, struct Node *item)
{
	(*item).next = head;
	return item;
}
