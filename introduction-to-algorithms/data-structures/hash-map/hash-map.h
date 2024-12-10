#include <stddef.h>

#ifndef HASH_MAP_H
#define HASH_MAP_H

const extern size_t HASH_MAP_DEFAULT_SIZE;
const extern int HASH_MAP_DATA_NOT_FOUND;
const extern int HASH_MAP_DATA_REMOVED;

struct HashMapData {
  int key;
  int data;
};
struct HashMapNode {
  struct HashMapData *data;
  struct HashMapNode *next;
  struct HashMapNode *prev;
};
struct HashMap {
  struct HashMapNode **_slots;
  size_t _current_size;
};

struct HashMap *HashMap();
void HashMap__del(struct HashMap *hm);
void HashMap__insert(struct HashMap *const hm, struct HashMapData *d);
void HashMap__remove(const struct HashMap *const hm, int key);
void HashMap__show(const struct HashMap *const hm);

struct HashMapNode *HashMapNode();
void HashMapNode__del(struct HashMapNode *const n);
void HashMapNode__show(const struct HashMapNode *const n);
struct HashMapNode *HashMapNode__prepend(struct HashMapNode *const n);
int HashMapNode__find(const struct HashMapNode *const n, int key,
                      const struct HashMapData *d);
struct HashMapNode *HashMapNode__remove(struct HashMapNode *n, int idx);

struct HashMapData *HashMapData(const int key, const int data);
void HashMapData__del(struct HashMapData *d);
int HashMapData__hash(const int k);
void HashMapData__show(const struct HashMapData *const d);
void HashMapData__init(struct HashMapData *d);

int HashMap__find(const struct HashMap *const hm, int key,
                  const struct HashMapData *d);

#endif
