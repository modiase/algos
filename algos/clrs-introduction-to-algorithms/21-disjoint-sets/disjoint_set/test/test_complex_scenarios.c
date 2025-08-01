#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "disjoint_set.h"

void test_complex_scenario();

int main()
{
    printf("Running Complex Scenarios Tests...\n");

    test_complex_scenario();
    printf("PASSED: test_complex_scenario\n");

    printf("\nAll complex scenarios tests passed successfully!\n");

    return EXIT_SUCCESS;
}

void test_complex_scenario()
{
    const int num_elements = 10;
    Node *elements[num_elements];

    for (int i = 0; i < num_elements; i++)
    {
        elements[i] = make_set(i);
        assert(find_set(elements[i])->size == 1);
    }

    union_sets(elements[0], elements[1]);
    assert(find_set(elements[0]) == find_set(elements[1]));
    assert(find_set(elements[0])->size == 2);

    union_sets(elements[2], elements[3]);
    assert(find_set(elements[2]) == find_set(elements[3]));
    assert(find_set(elements[2])->size == 2);
    assert(find_set(elements[0]) != find_set(elements[2]));

    Node *rep_set1 = find_set(elements[0]);
    Node *rep_set2 = find_set(elements[2]);
    Node *big_rep = union_sets(elements[0], elements[2]);
    assert(big_rep == rep_set1 || big_rep == rep_set2);
    assert(big_rep->size == 4);
    assert(find_set(elements[0]) == big_rep);
    assert(find_set(elements[1]) == big_rep);
    assert(find_set(elements[2]) == big_rep);
    assert(find_set(elements[3]) == big_rep);

    union_sets(elements[5], elements[6]);
    union_sets(elements[7], elements[8]);
    Node *other_big_rep = union_sets(elements[5], elements[7]);
    assert(other_big_rep->size == 4);
    assert(find_set(elements[5]) == other_big_rep);
    assert(find_set(elements[6]) == other_big_rep);
    assert(find_set(elements[7]) == other_big_rep);
    assert(find_set(elements[8]) == other_big_rep);

    Node *final_rep = union_sets(elements[1], elements[8]);
    assert(final_rep->size == 8);
    assert(find_set(elements[3]) == final_rep);
    assert(find_set(elements[5]) == final_rep);

    assert(find_set(elements[4])->size == 1);
    assert(find_set(elements[9])->size == 1);
    assert(find_set(elements[4]) != final_rep);
    assert(find_set(elements[9]) != final_rep);

    destroy_set(final_rep);
    destroy_set(find_set(elements[4]));
    destroy_set(find_set(elements[9]));
}
