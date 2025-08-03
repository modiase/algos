#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "disjoint_set.h"

void test_basic_union();
void test_union_by_size();
void test_idempotent_union();

int main()
{
    printf("Running Union Operations Tests...\n");

    test_basic_union();
    printf("PASSED: test_basic_union\n");

    test_union_by_size();
    printf("PASSED: test_union_by_size\n");

    test_idempotent_union();
    printf("PASSED: test_idempotent_union\n");

    printf("\nAll union operations tests passed successfully!\n");

    return EXIT_SUCCESS;
}

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

void test_union_by_size()
{
    Node *n1 = make_set(100);
    Node *n2 = make_set(101);
    Node *rep_large = union_sets(n1, n2);
    assert(rep_large->size == 2);

    Node *n3 = make_set(200);
    Node *rep_small_original = find_set(n3);
    assert(rep_small_original->size == 1);

    Node *final_rep = union_sets(n3, n1);
    assert(final_rep == rep_large);
    assert(final_rep->size == 3);

    assert(find_set(n1) == rep_large);
    assert(find_set(n2) == rep_large);
    assert(find_set(n3) == rep_large);

    destroy_set(final_rep);
}

void test_idempotent_union()
{
    Node *n1 = make_set(40);
    Node *n2 = make_set(50);

    Node *rep1 = union_sets(n1, n2);
    assert(rep1->size == 2);

    Node *rep2 = union_sets(n1, n2);

    assert(rep1 == rep2);
    assert(rep1->size == 2);

    destroy_set(rep1);
}
