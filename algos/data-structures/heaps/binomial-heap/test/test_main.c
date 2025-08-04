#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <limits.h>
#include <stdbool.h>

#include "binomial-heap.h"

static int tests_run = 0;
static int tests_passed = 0;

static void run_test(const char *test_name, bool (*test_func)(void))
{
    tests_run++;
    printf("Running %s... ", test_name);
    fflush(stdout);

    if (test_func()) {
        tests_passed++;
        printf("PASSED\n");
    } else {
        printf("FAILED\n");
    }
}

bool test_heap_creation_and_destruction(void)
{
    struct binomial_heap *heap = binomial_heap_create();
    if (!heap)
        return false;

    bool is_empty = binomial_heap_is_empty(heap);
    int size = binomial_heap_size(heap);

    binomial_heap_destroy(heap);

    return is_empty && size == 0;
}

bool test_single_insert_and_minimum(void)
{
    struct binomial_heap *heap = binomial_heap_create();
    if (!heap)
        return false;

    bool insert_success = binomial_heap_insert(heap, 42);
    int minimum = binomial_heap_minimum(heap);
    bool not_empty = !binomial_heap_is_empty(heap);
    int size = binomial_heap_size(heap);

    binomial_heap_destroy(heap);

    return insert_success && minimum == 42 && not_empty && size == 1;
}

bool test_multiple_inserts_maintain_minimum(void)
{
    struct binomial_heap *heap = binomial_heap_create();
    if (!heap)
        return false;

    int values[] = {10, 5, 15, 3, 8, 12, 1, 20};
    int num_values = sizeof(values) / sizeof(values[0]);

    for (int i = 0; i < num_values; i++) {
        if (!binomial_heap_insert(heap, values[i])) {
            binomial_heap_destroy(heap);
            return false;
        }
    }

    int minimum = binomial_heap_minimum(heap);
    int size = binomial_heap_size(heap);

    binomial_heap_destroy(heap);

    return minimum == 1 && size == num_values;
}

bool test_extract_minimum_single_element(void)
{
    struct binomial_heap *heap = binomial_heap_create();
    if (!heap)
        return false;

    binomial_heap_insert(heap, 100);
    int extracted = binomial_heap_extract_min(heap);
    bool is_empty = binomial_heap_is_empty(heap);
    int size = binomial_heap_size(heap);

    binomial_heap_destroy(heap);

    return extracted == 100 && is_empty && size == 0;
}

bool test_extract_minimum_multiple_elements(void)
{
    struct binomial_heap *heap = binomial_heap_create();
    if (!heap)
        return false;

    int values[] = {50, 30, 70, 20, 40, 60, 10};
    int expected_order[] = {10, 20, 30, 40, 50, 60, 70};
    int num_values = sizeof(values) / sizeof(values[0]);

    for (int i = 0; i < num_values; i++) {
        binomial_heap_insert(heap, values[i]);
    }

    bool correct_order = true;
    for (int i = 0; i < num_values; i++) {
        int extracted = binomial_heap_extract_min(heap);
        if (extracted != expected_order[i]) {
            correct_order = false;
            break;
        }
    }

    bool is_empty = binomial_heap_is_empty(heap);
    binomial_heap_destroy(heap);

    return correct_order && is_empty;
}

bool test_heap_union_operation(void)
{
    struct binomial_heap *heap1 = binomial_heap_create();
    struct binomial_heap *heap2 = binomial_heap_create();
    if (!heap1 || !heap2)
        return false;

    binomial_heap_insert(heap1, 5);
    binomial_heap_insert(heap1, 15);
    binomial_heap_insert(heap1, 25);

    binomial_heap_insert(heap2, 10);
    binomial_heap_insert(heap2, 20);

    struct binomial_heap *merged = binomial_heap_union(heap1, heap2);
    if (!merged)
        return false;

    int size = binomial_heap_size(merged);
    int minimum = binomial_heap_minimum(merged);

    int expected_order[] = {5, 10, 15, 20, 25};
    bool correct_order = true;
    for (int i = 0; i < 5; i++) {
        int extracted = binomial_heap_extract_min(merged);
        if (extracted != expected_order[i]) {
            correct_order = false;
            break;
        }
    }

    binomial_heap_destroy(merged);

    return size == 5 && minimum == 5 && correct_order;
}

bool test_empty_heap_operations(void)
{
    struct binomial_heap *heap = binomial_heap_create();
    if (!heap)
        return false;

    int minimum = binomial_heap_minimum(heap);
    int extracted = binomial_heap_extract_min(heap);
    bool is_empty = binomial_heap_is_empty(heap);
    int size = binomial_heap_size(heap);

    binomial_heap_destroy(heap);

    return minimum == INT_MAX && extracted == INT_MAX && is_empty && size == 0;
}

bool test_large_dataset_operations(void)
{
    struct binomial_heap *heap = binomial_heap_create();
    if (!heap)
        return false;

    const int num_elements = 1000;

    for (int i = num_elements; i > 0; i--) {
        if (!binomial_heap_insert(heap, i)) {
            binomial_heap_destroy(heap);
            return false;
        }
    }

    int size = binomial_heap_size(heap);
    int minimum = binomial_heap_minimum(heap);

    bool correct_extraction = true;
    for (int i = 1; i <= num_elements; i++) {
        int extracted = binomial_heap_extract_min(heap);
        if (extracted != i) {
            correct_extraction = false;
            break;
        }
    }

    bool final_empty = binomial_heap_is_empty(heap);
    binomial_heap_destroy(heap);

    return size == num_elements && minimum == 1 && correct_extraction && final_empty;
}

bool test_duplicate_values_handling(void)
{
    struct binomial_heap *heap = binomial_heap_create();
    if (!heap)
        return false;

    int values[] = {5, 3, 5, 1, 3, 1, 5};
    int expected_order[] = {1, 1, 3, 3, 5, 5, 5};
    int num_values = sizeof(values) / sizeof(values[0]);

    for (int i = 0; i < num_values; i++) {
        binomial_heap_insert(heap, values[i]);
    }

    bool correct_order = true;
    for (int i = 0; i < num_values; i++) {
        int extracted = binomial_heap_extract_min(heap);
        if (extracted != expected_order[i]) {
            correct_order = false;
            break;
        }
    }

    binomial_heap_destroy(heap);
    return correct_order;
}

bool test_alternating_insert_extract(void)
{
    struct binomial_heap *heap = binomial_heap_create();
    if (!heap)
        return false;

    binomial_heap_insert(heap, 10);
    binomial_heap_insert(heap, 5);

    int first_min = binomial_heap_extract_min(heap);

    binomial_heap_insert(heap, 3);
    binomial_heap_insert(heap, 15);

    int second_min = binomial_heap_extract_min(heap);
    int third_min = binomial_heap_extract_min(heap);
    int fourth_min = binomial_heap_extract_min(heap);

    bool final_empty = binomial_heap_is_empty(heap);

    binomial_heap_destroy(heap);

    return first_min == 5 && second_min == 3 && third_min == 10 &&
           fourth_min == 15 && final_empty;
}

bool test_union_with_empty_heaps(void)
{
    struct binomial_heap *heap1 = binomial_heap_create();
    struct binomial_heap *heap2 = binomial_heap_create();
    struct binomial_heap *heap3 = binomial_heap_create();
    if (!heap1 || !heap2 || !heap3)
        return false;

    binomial_heap_insert(heap1, 42);

    struct binomial_heap *merged1 = binomial_heap_union(heap1, heap2);
    struct binomial_heap *merged2 = binomial_heap_union(heap3, merged1);

    if (!merged1 || !merged2)
        return false;

    int size = binomial_heap_size(merged2);
    int minimum = binomial_heap_minimum(merged2);
    int extracted = binomial_heap_extract_min(merged2);
    bool final_empty = binomial_heap_is_empty(merged2);

    binomial_heap_destroy(merged2);

    return size == 1 && minimum == 42 && extracted == 42 && final_empty;
}

bool test_stress_mixed_operations(void)
{
    struct binomial_heap *heap1 = binomial_heap_create();
    struct binomial_heap *heap2 = binomial_heap_create();
    if (!heap1 || !heap2)
        return false;

    for (int i = 100; i >= 1; i -= 2) {
        binomial_heap_insert(heap1, i);
    }

    for (int i = 99; i >= 1; i -= 2) {
        binomial_heap_insert(heap2, i);
    }

    struct binomial_heap *merged = binomial_heap_union(heap1, heap2);
    if (!merged)
        return false;

    int size = binomial_heap_size(merged);

    for (int i = 0; i < 25; i++) {
        int min_before = binomial_heap_minimum(merged);
        int extracted = binomial_heap_extract_min(merged);
        if (min_before != extracted || extracted != i + 1) {
            binomial_heap_destroy(merged);
            return false;
        }
    }

    int remaining_size = binomial_heap_size(merged);

    binomial_heap_destroy(merged);

    return size == 100 && remaining_size == 75;
}

int main(void)
{
    printf("Running Binomial Heap Test Suite\n");
    printf("================================\n\n");

    run_test("Heap Creation and Destruction", test_heap_creation_and_destruction);
    run_test("Single Insert and Minimum", test_single_insert_and_minimum);
    run_test("Multiple Inserts Maintain Minimum", test_multiple_inserts_maintain_minimum);
    run_test("Extract Minimum Single Element", test_extract_minimum_single_element);
    run_test("Extract Minimum Multiple Elements", test_extract_minimum_multiple_elements);
    run_test("Heap Union Operation", test_heap_union_operation);
    run_test("Empty Heap Operations", test_empty_heap_operations);
    run_test("Large Dataset Operations", test_large_dataset_operations);
    run_test("Duplicate Values Handling", test_duplicate_values_handling);
    run_test("Alternating Insert/Extract", test_alternating_insert_extract);
    run_test("Union with Empty Heaps", test_union_with_empty_heaps);
    run_test("Stress Mixed Operations", test_stress_mixed_operations);

    printf("\n================================\n");
    printf("Test Results: %d/%d tests passed\n", tests_passed, tests_run);

    if (tests_passed == tests_run) {
        printf("All tests passed! ✓\n");
        return EXIT_SUCCESS;
    } else {
        printf("Some tests failed! ✗\n");
        return EXIT_FAILURE;
    }
}
