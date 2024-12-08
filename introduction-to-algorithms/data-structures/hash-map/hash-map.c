#include <stdio.h>
#include <stdlib.h>

#include "hash-map.h"

const size_t HASH_MAP_DEFAULT_SIZE = 16;

struct HashMapData *HashMapData();
void HashMapData__del(struct HashMapData *d);
int HashMapData__hash(struct HashMapData *d);
void HashMapData__show(struct HashMapData *d){};

struct HashMapNode *HashMapNode() {
  struct HashMapNode *n =
      (struct HashMapNode *)malloc(sizeof(struct HashMapNode));
  n->next = NULL;
  n->prev = NULL;
  n->data = NULL;
  return n;
}

void HashMapNode__del(struct HashMapNode *n) {
  if (n == NULL)
    return;
  if (n->data != NULL)
    free(n->data);
  free(n);
}
void HashMapNode__show(struct HashMapNode *n) {
  puts("<HashMapNode>");
  if (n->data != NULL)
    HashMapData__show(n->data);
  puts("</HashMapNode>");
  if (n->next != NULL)
    HashMapNode__show(n->next);
}

struct HashMap *HashMap() {
  struct HashMap *hm = (struct HashMap *)malloc(sizeof(struct HashMap));
  hm->_slots = (struct HashMapNode **)malloc(HASH_MAP_DEFAULT_SIZE *
                                             sizeof(struct HashMapNode *));
  hm->_current_size = HASH_MAP_DEFAULT_SIZE;
  for (size_t i = 0; i < hm->_current_size; i++) {
    *(hm->_slots + i) = HashMapNode();
  }
  return hm;
}

void HashMap__del(struct HashMap *hm) {
  for (size_t i = 0; i < hm->_current_size; i++) {
    HashMapNode__del(*(hm->_slots + i));
  }
  free(hm->_slots);
  free(hm);
}

void HashMap__show(struct HashMap *hm) {
  puts("<HashMap>");
  for (size_t i = 0; i < hm->_current_size; i++) {
    puts("<HashMapSlot>");
    HashMapNode__show(*(hm->_slots + i));
    puts("</HashMapSlot>");
  }
  puts("</HashMap>");
}
