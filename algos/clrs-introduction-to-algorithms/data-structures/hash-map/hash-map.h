/* TODO: Use an enum to return errors
 */
#include <stddef.h>

#ifndef HASH_MAP_H
#define HASH_MAP_H

const extern size_t HASH_MAP_DEFAULT_SIZE;
const extern int HASH_MAP_NODE_SENTINEL_KEY;

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

struct HashMap2 {
    struct HashMapData *_slotsHead;
    size_t _current_size;
};

struct HashMap2 *HashMap2();
void HashMap2__del(struct HashMap2 *hm);
void HashMap2__insert(struct HashMap2 *const hm, struct HashMapData *d);
void HashMap2__remove(const struct HashMap2 *const hm, int key);
void HashMap2__show(const struct HashMap2 *const hm);

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
int HashMapNode__size(const struct HashMapNode *const n);
struct HashMapNode *HashMapNode__remove(struct HashMapNode *n, int idx);

struct HashMapData *HashMapData(const int key, const int data);
void HashMapData__del(struct HashMapData *d);
int HashMapData__hash(const int k);
int HashMapData__hash2(const int k, const int i);
int _HashMapData__hash2(const int k);
void HashMapData__show(const struct HashMapData *const d);
void HashMapData__init(struct HashMapData *d);

int HashMap__find(const struct HashMap *const hm, int key,
                  const struct HashMapData *d);

#endif
