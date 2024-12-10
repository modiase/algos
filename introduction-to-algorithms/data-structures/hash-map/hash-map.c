#include <stdio.h>
#include <stdlib.h>

#include "hash-map.h"

const size_t HASH_MAP_DEFAULT_SIZE = 16;
const int HASH_MAP_DATA_NOT_FOUND = -1;
const int HASH_MAP_DATA_REMOVED = 1;
const char TAB[] = "    ";
#define INDENT_WIDTH 4
#define INDENT(n)                                                              \
  do {                                                                         \
    for (size_t _i = 0; _i < (n); _i++) {                                      \
      printf("%*s", INDENT_WIDTH, "");                                         \
    }                                                                          \
  } while (0)

void HashMapData__init(struct HashMapData *d) {
  d->data = 0;
  d->key = 0;
};

struct HashMapData *HashMapData(const int key, const int data) {
  struct HashMapData *d =
      (struct HashMapData *)malloc(sizeof(struct HashMapData));
  d->key = key;
  d->data = data;
  return d;
};

void HashMapData__del(struct HashMapData *d) {
  if (d != NULL)
    return;
  free(d);
  d = NULL;
};
int HashMapData__hash(const int key) { return key % HASH_MAP_DEFAULT_SIZE; };
void HashMapData__show(const struct HashMapData *const d) {
  INDENT(3);
  printf("<HashMapData key=%d, data=%d />\n", d->key, d->data);
};

void HashMapNode__init(struct HashMapNode *const n) {
  n->next = NULL;
  n->prev = NULL;
  n->data = NULL;
}

struct HashMapNode *HashMapNode() {
  struct HashMapNode *n =
      (struct HashMapNode *)malloc(sizeof(struct HashMapNode));
  HashMapNode__init(n);
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
  HashMapData__del(n->data);
  free(n);
  n = NULL;
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

struct HashMapNode *HashMapNode__remove(struct HashMapNode *n, int idx) {
  if (n == NULL)
    return n;

  if (idx == 0) {
    struct HashMapNode *new_head = n->next;
    if (new_head != NULL)
      new_head->prev = NULL;
    HashMapNode__del(n);
    return new_head;
  }

  int c = idx;
  struct HashMapNode *current = n;
  while (c > 0) {
    if (current->next == NULL)
      return n;
    current = current->next;
    c--;
  }
  current->prev->next = current->next;
  if (current->next != NULL)
    current->next->prev = current->prev;
  HashMapNode__del(current);
  return n;
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
  if (hm->_slots != NULL)
    free(hm->_slots);
  hm->_slots = NULL;
  free(hm);
  hm = NULL;
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
  struct HashMapNode **slot = hm->_slots + h;
  return HashMapNode__find(*slot, key, d);
}

void HashMap__remove(const struct HashMap *const hm, int key) {
  struct HashMapData *d;
  size_t h = HashMapData__hash(key);
  struct HashMapNode **slot = hm->_slots + h;
  *slot = HashMapNode__remove(*slot, HashMapNode__find(*slot, key, d));
}
