#include <stdio.h>
#include <stdlib.h>

#include "hash-map.h"

const size_t HASH_MAP_DEFAULT_SIZE = 16;
const int HASH_MAP_DATA_NOT_FOUND = -1;
const char TAB[] = "    ";
#define INDENT_WIDTH 4
#define INDENT(n)                                                              \
  do {                                                                         \
    for (size_t _i = 0; _i < (n); _i++) {                                      \
      printf("%*s", INDENT_WIDTH, "");                                         \
    }                                                                          \
  } while (0)

struct HashMapData *HashMapData(const int key, const int data) {
  struct HashMapData *d =
      (struct HashMapData *)malloc(sizeof(struct HashMapData));
  d->key = key;
  d->data = data;
  return d;
};

void HashMapData__del(struct HashMapData *d) { free(d); };
int HashMapData__hash(const int key) {
  // Naive hash function
  return key % HASH_MAP_DEFAULT_SIZE;
};
void HashMapData__show(const struct HashMapData *const d) {
  INDENT(3);
  printf("<HashMapData key=%d, data=%d />\n", d->key, d->data);
};

struct HashMapNode *HashMapNode() {
  struct HashMapNode *n =
      (struct HashMapNode *)malloc(sizeof(struct HashMapNode));
  n->next = NULL;
  n->prev = NULL;
  n->data = NULL;
  return n;
}

struct HashMapNode *HashMapNode__prepend(struct HashMapNode *const n) {
  struct HashMapNode *prepended = HashMapNode();
  n->prev = prepended;
  prepended->next = n;
  return prepended;
}

void HashMapNode__del(struct HashMapNode *n) {
  if (n == NULL)
    return;
  if (n->data != NULL)
    free(n->data);
  free(n);
}

int HashMapNode__find(const struct HashMapNode *const n, int key,
                      const struct HashMapData *d) {
  if (n == NULL)
    return HASH_MAP_DATA_NOT_FOUND;
  const struct HashMapNode *current = n;
  int i = 0;
  do {
    if (current->data->key == key) {
      d = current->data;
      return i;
    }
    current = current->next;
    i++;
  } while (current != NULL);
  return HASH_MAP_DATA_NOT_FOUND;
}

void HashMapNode__show(const struct HashMapNode *const n) {
  if (n == NULL)
    return;
  INDENT(2);
  puts("<HashMapNode>");
  if (n->data != NULL)
    HashMapData__show(n->data);
  INDENT(2);
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
    *(hm->_slots + i) = NULL;
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

void HashMap__show(const struct HashMap *const hm) {
  struct HashMapNode **slot;
  puts("<HashMap>");
  for (size_t i = 0; i < hm->_current_size; i++) {
    slot = (hm->_slots + i);
    if (*slot == NULL)
      continue;
    INDENT(1);
    printf("<HashMapSlot idx=%ld slot=%p>\n", i, hm->_slots + i);
    HashMapNode__show(*slot);
    INDENT(1);
    puts("</HashMapSlot>");
  }
  puts("</HashMap>");
}

void HashMap__insert(struct HashMap *const hm, struct HashMapData *d) {
  const int h = (size_t)HashMapData__hash(d->key);
  struct HashMapNode **slot = hm->_slots + h;
  struct HashMapNode *prependedNode;
  if (*slot == NULL)
    prependedNode = HashMapNode();
  else
    prependedNode = HashMapNode__prepend(*slot);
  *slot = prependedNode;
  prependedNode->data = d;
}

int HashMap__find(const struct HashMap *const hm, int key,
                  const struct HashMapData *d) {
  size_t h = HashMapData__hash(key);
  printf("h=%ld\n", h);
  struct HashMapNode **slot = hm->_slots + h;
  return HashMapNode__find(*slot, key, d);
}
