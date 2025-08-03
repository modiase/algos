#include <stdlib.h>
#include <stdio.h>
#include "linked-list.h"

int main()
{
    void *MyObj = malloc(10);
    struct Node n1 = (struct Node) {
        .data = MyObj,.next = &NIL
    };
    struct Node n2 = (struct Node) {
        .data = MyObj,.next = &NIL
    };

    struct Node *MyList = LinkedList__prepend(&n2, &n1);
    printf("%p\n", n1.next);
    printf("%p\n", &n2);
    printf("%p\n", &n1);
    printf("%p\n", MyList);

    return 0;
}
