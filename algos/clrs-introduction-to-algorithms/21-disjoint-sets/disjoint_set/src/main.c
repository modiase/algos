#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "disjoint_set.h"

void print_sets(Node * nodes[], int count);

int main(void)
{
    unsigned int seed;
    char *seed_env = getenv("SEED");

    if (seed_env != NULL) {
        seed = (unsigned int)atoi(seed_env);
    } else {
        seed = (unsigned int)time(NULL);
    }

    printf("Using seed: %u\n", seed);
    srand(seed);

    int values[10];
    Node *nodes[10];

    printf("Generated integers: ");
    for (int i = 0; i < 10; i++) {
        values[i] = rand() % 101;
        printf("%d ", values[i]);
        nodes[i] = make_set(values[i]);
    }
    printf("\n\n");

    int union_count = (rand() % 10) + 1;
    printf("Performing %d random union operations:\n", union_count);

    for (int i = 0; i < union_count; i++) {
        int idx1 = rand() % 10;
        int idx2 = rand() % 10;
        if (idx1 != idx2) {
            Node *rep1 = find_set(nodes[idx1]);
            Node *rep2 = find_set(nodes[idx2]);
            if (rep1 != rep2) {
                printf("Union(%d, %d)\n", values[idx1],
                       values[idx2]);
                union_sets(nodes[idx1], nodes[idx2]);
            }
        }
    }

    printf("\nFinal disjoint sets:\n");
    print_sets(nodes, 10);

    Node *representatives[10];
    int rep_count = 0;

    for (int i = 0; i < 10; i++) {
        Node *rep = find_set(nodes[i]);
        int found = 0;
        for (int j = 0; j < rep_count; j++) {
            if (representatives[j] == rep) {
                found = 1;
                break;
            }
        }
        if (!found) {
            representatives[rep_count++] = rep;
        }
    }

    for (int i = 0; i < rep_count; i++) {
        destroy_set(representatives[i]);
    }

    return EXIT_SUCCESS;
}

void print_sets(Node *nodes[], int count)
{
    Node *representatives[10];
    int rep_count = 0;

    for (int i = 0; i < count; i++) {
        Node *rep = find_set(nodes[i]);
        int found = 0;
        for (int j = 0; j < rep_count; j++) {
            if (representatives[j] == rep) {
                found = 1;
                break;
            }
        }
        if (!found) {
            representatives[rep_count++] = rep;
        }
    }

    for (int set_idx = 0; set_idx < rep_count; set_idx++) {
        Node *rep = representatives[set_idx];
        printf("Set %d (size %d): { ", set_idx + 1, rep->size);

        Node *current = rep->next;
        while (current != rep) {
            printf("%d ", current->key);
            current = current->next;
        }
        printf("}\n");
    }
}
