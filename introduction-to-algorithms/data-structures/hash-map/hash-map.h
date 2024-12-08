#include <stddef.h>

#ifndef HASH_MAP_H
#define HASH_MAP_H

const extern size_t HASH_MAP_DEFAULT_SIZE;

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
void HashMap__insert(struct HashMap *hm, struct HashMapData *d);
void HashMap__remove(struct HashMap *hm, int key);
void HashMap__show(struct HashMap *hm);

struct HashMapNode *HashMapNode();
void HashMapNode__del(struct HashMapNode *n);
void HashMapNode__show(struct HashMapNode *n);

struct HashMapData *HashMapData();
void HashMapData__del(struct HashMapData *d);
int HashMapData__hash(struct HashMapData *d);
void HashMapData__show(struct HashMapData *d);

#endif
