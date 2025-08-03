#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "disjoint_set.h"

void test_make_set_and_find_set();

int main()
{
	printf("Running Basic Operations Tests...\n");

	test_make_set_and_find_set();
	printf("PASSED: test_make_set_and_find_set\n");

	printf("\nAll basic operations tests passed successfully!\n");

	return EXIT_SUCCESS;
}

void test_make_set_and_find_set()
{
	Node *n1 = make_set(10);
	assert(n1 != NULL);
	assert(n1->key == 10);

	Node *rep1 = find_set(n1);
	assert(rep1 != NULL);
	assert(rep1->size == 1);
	assert(rep1 == n1->representative);
	assert(rep1->representative == rep1);

	assert(rep1->next == n1);
	assert(rep1->prev == n1);
	assert(n1->next == rep1);
	assert(n1->prev == rep1);

	destroy_set(rep1);
}
