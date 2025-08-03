#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "disjoint_set.h"

void test_make_set_and_find_set();
void test_basic_union();
void test_union_by_size();
void test_idempotent_union();
void test_complex_scenario();

int main()
{
	printf("Running Disjoint Set Tests...\n");

	test_make_set_and_find_set();
	printf("PASSED: test_make_set_and_find_set\n");

	test_basic_union();
	printf("PASSED: test_basic_union\n");

	test_union_by_size();
	printf("PASSED: test_union_by_size\n");

	test_idempotent_union();
	printf("PASSED: test_idempotent_union\n");

	test_complex_scenario();
	printf("PASSED: test_complex_scenario\n");

	printf("\nAll tests passed successfully!\n");

	return EXIT_SUCCESS;
}

/**
 * @brief Tests the creation of a single set and finding its representative.
 */
void test_make_set_and_find_set()
{
	Node *n1 = make_set(10);
	assert(n1 != NULL);
	assert(n1->key == 10);

	Node *rep1 = find_set(n1);
	assert(rep1 != NULL);
	assert(rep1->size == 1);
	assert(rep1 == n1->representative);
	assert(rep1->representative == rep1);	// Representative points to itself.

	// The single data node and head node should point to each other.
	assert(rep1->next == n1);
	assert(rep1->prev == n1);
	assert(n1->next == rep1);
	assert(n1->prev == rep1);

	destroy_set(rep1);
}

/**
 * @brief Tests a simple union of two sets of size 1.
 */
void test_basic_union()
{
	Node *n1 = make_set(20);
	Node *n2 = make_set(30);

	Node *rep1_before = find_set(n1);
	Node *rep2_before = find_set(n2);
	assert(rep1_before != rep2_before);
	assert(rep1_before->size == 1);
	assert(rep2_before->size == 1);

	Node *new_rep = union_sets(n1, n2);
	assert(new_rep != NULL);
	assert(find_set(n1) == new_rep);
	assert(find_set(n2) == new_rep);
	assert(new_rep->size == 2);

	destroy_set(new_rep);
}

/**
 * @brief Tests the union-by-size heuristic by merging a smaller set into a larger one.
 */
void test_union_by_size()
{
	// Create a larger set of size 2
	Node *n1 = make_set(100);
	Node *n2 = make_set(101);
	Node *rep_large = union_sets(n1, n2);
	assert(rep_large->size == 2);

	// Create a smaller set of size 1
	Node *n3 = make_set(200);
	Node *rep_small_original = find_set(n3);
	assert(rep_small_original->size == 1);

	// Union the smaller set into the larger one.
	// The representative of the larger set should be the new representative.
	Node *final_rep = union_sets(n3, n1);
	assert(final_rep == rep_large);
	assert(final_rep->size == 3);

	// Check that all nodes now point to the same representative
	assert(find_set(n1) == rep_large);
	assert(find_set(n2) == rep_large);
	assert(find_set(n3) == rep_large);

	destroy_set(final_rep);
}

/**
 * @brief Tests that performing a union on two elements already in the same set has no effect.
 */
void test_idempotent_union()
{
	Node *n1 = make_set(40);
	Node *n2 = make_set(50);

	Node *rep1 = union_sets(n1, n2);
	assert(rep1->size == 2);

	// Perform the union again
	Node *rep2 = union_sets(n1, n2);

	// The representative and size should be unchanged
	assert(rep1 == rep2);
	assert(rep1->size == 2);

	destroy_set(rep1);
}

/**
 * @brief Tests a more complex sequence of make_set and union operations.
 */
void test_complex_scenario()
{
	const int num_elements = 10;
	Node *elements[num_elements];

	// Create 10 disjoint sets for elements 0 through 9
	for (int i = 0; i < num_elements; i++) {
		elements[i] = make_set(i);
		assert(find_set(elements[i])->size == 1);
	}

	// Union(0, 1) -> {0, 1}
	union_sets(elements[0], elements[1]);
	assert(find_set(elements[0]) == find_set(elements[1]));
	assert(find_set(elements[0])->size == 2);

	// Union(2, 3) -> {2, 3}
	union_sets(elements[2], elements[3]);
	assert(find_set(elements[2]) == find_set(elements[3]));
	assert(find_set(elements[2])->size == 2);
	assert(find_set(elements[0]) != find_set(elements[2]));	// Still separate

	// Union(0, 2) -> {0, 1, 2, 3}
	Node *rep_set1 = find_set(elements[0]);
	Node *rep_set2 = find_set(elements[2]);
	Node *big_rep = union_sets(elements[0], elements[2]);
	assert(big_rep == rep_set1 || big_rep == rep_set2);	// One of the old reps becomes the new
	assert(big_rep->size == 4);
	assert(find_set(elements[0]) == big_rep);
	assert(find_set(elements[1]) == big_rep);
	assert(find_set(elements[2]) == big_rep);
	assert(find_set(elements[3]) == big_rep);

	// Union(5, 6), Union(7, 8), Union(5, 7) -> {5, 6, 7, 8}
	union_sets(elements[5], elements[6]);
	union_sets(elements[7], elements[8]);
	Node *other_big_rep = union_sets(elements[5], elements[7]);
	assert(other_big_rep->size == 4);
	assert(find_set(elements[5]) == other_big_rep);
	assert(find_set(elements[6]) == other_big_rep);
	assert(find_set(elements[7]) == other_big_rep);
	assert(find_set(elements[8]) == other_big_rep);

	// Union the two big sets -> {0, 1, 2, 3, 5, 6, 7, 8}
	Node *final_rep = union_sets(elements[1], elements[8]);
	assert(final_rep->size == 8);
	assert(find_set(elements[3]) == final_rep);
	assert(find_set(elements[5]) == final_rep);

	// Element 4 and 9 are still in their own sets
	assert(find_set(elements[4])->size == 1);
	assert(find_set(elements[9])->size == 1);
	assert(find_set(elements[4]) != final_rep);
	assert(find_set(elements[9]) != final_rep);

	// Cleanup memory
	destroy_set(final_rep);
	destroy_set(find_set(elements[4]));
	destroy_set(find_set(elements[9]));
}
