#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "hash-map.h"

const size_t HASH_MAP_DEFAULT_SIZE = 771;
const float A = 0.6180339887;
const float A_2 = 1 - (1 / 0.6180339887);
const int HASH_MAP_DATA_NOT_FOUND = -1;
const int HASH_MAP_DATA_REMOVED = 1;
const int HASH_MAP_NODE_SENTINEL_KEY = 1;
const int HASH_MAP_SLOT_FULL = -1;
const char TAB[] = "    ";
#define INDENT_WIDTH 4
#define INDENT(n)                                                              \
  do {                                                                         \
    for (size_t _i = 0; _i < (n); _i++) {                                      \
      printf("%*s", INDENT_WIDTH, "");                                         \
    }                                                                          \
  } while (0)

void HashMapData__init(struct HashMapData *d)
{
	d->data = 0;
	d->key = 0;
};

struct HashMapData *HashMapData(const int key, const int data)
{
	struct HashMapData *d =
	    (struct HashMapData *)malloc(sizeof(struct HashMapData));
	d->key = key;
	d->data = data;
	return d;
};

void HashMapData__del(struct HashMapData *d)
{
	if (d != NULL)
		return;
	free(d);
	d = NULL;
};

int HashMapData__hash(const int key)
{
	return floor(fmod(key * A, 1.0) * HASH_MAP_DEFAULT_SIZE);
};

int _HashMapData2__hash(const int key)
{
	return floor(fmod(key * A_2, 1.0) * HASH_MAP_DEFAULT_SIZE);
};

int HashMapData2__hash(const int key, const int i, const size_t m)
{
	return (HashMapData__hash(key) + i * _HashMapData2__hash(key)) % m;
};

void HashMapData__show(const struct HashMapData *const d)
{
	INDENT(3);
	printf("<HashMapData key=%d, data=%d />\n", d->key, d->data);
};

void HashMapData__makeNil(struct HashMapData *const d)
{
	if (d == NULL)
		return;
	d->key = HASH_MAP_NODE_SENTINEL_KEY;
}

bool HashMapData__isNil(const struct HashMapData *const d)
{
	if (d == NULL)
		return false;
	return d->key == HASH_MAP_NODE_SENTINEL_KEY;
}

void HashMapNode__init(struct HashMapNode *const n)
{
	n->next = NULL;
	n->prev = NULL;
	n->data = NULL;
}

struct HashMapNode *HashMapNode()
{
	struct HashMapNode *n =
	    (struct HashMapNode *)malloc(sizeof(struct HashMapNode));
	HashMapNode__init(n);
	return n;
}

struct HashMapNode *HashMapNode__prepend(struct HashMapNode *const n)
{
	struct HashMapNode *prepended = HashMapNode();
	n->prev = prepended;
	prepended->next = n;
	return prepended;
}

void HashMapNode__del(struct HashMapNode *n)
{
	if (n == NULL)
		return;
	HashMapData__del(n->data);
	free(n);
	n = NULL;
}

int HashMapNode__find(const struct HashMapNode *const n, int key,
		      const struct HashMapData *d)
{
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

int HashMapNode__size(const struct HashMapNode *const n)
{
	int i = 0;
	if (n == NULL)
		return i;
	const struct HashMapNode *current = n;
	do {
		i++;
		current = current->next;
	} while (current != NULL);
	return i;
}

struct HashMapNode *HashMapNode__remove(struct HashMapNode *n, int idx)
{
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

void HashMapNode__show(const struct HashMapNode *const n)
{
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

struct HashMap *HashMap()
{
	struct HashMap *hm = (struct HashMap *)malloc(sizeof(struct HashMap));
	hm->_slots = (struct HashMapNode **)malloc(HASH_MAP_DEFAULT_SIZE *
						   sizeof(struct HashMapNode
							  *));
	hm->_current_size = HASH_MAP_DEFAULT_SIZE;
	for (size_t i = 0; i < hm->_current_size; i++) {
		*(hm->_slots + i) = NULL;
	}
	return hm;
}

void HashMap__del(struct HashMap *hm)
{
	for (size_t i = 0; i < hm->_current_size; i++) {
		HashMapNode__del(*(hm->_slots + i));
	}
	if (hm->_slots != NULL)
		free(hm->_slots);
	hm->_slots = NULL;
	free(hm);
	hm = NULL;
}

void HashMap__show(const struct HashMap *const hm)
{
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

void HashMap__insert(struct HashMap *const hm, struct HashMapData *d)
{
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
		  const struct HashMapData *d)
{
	size_t h = HashMapData__hash(key);
	struct HashMapNode **slot = hm->_slots + h;
	return HashMapNode__find(*slot, key, d);
}

void HashMap__remove(const struct HashMap *const hm, int key)
{
	struct HashMapData *d;
	size_t h = HashMapData__hash(key);
	struct HashMapNode **slot = hm->_slots + h;
	*slot = HashMapNode__remove(*slot, HashMapNode__find(*slot, key, d));
}

void HashMap2__show(const struct HashMap2 *const hm)
{
	puts("<HashMap>");
	for (size_t i = 0; i < hm->_current_size; i++) {
		struct HashMapData *d = hm->_slotsHead + i;
		if (HashMapData__isNil(d))
			continue;
		INDENT(1);
		HashMapData__show(d);
		INDENT(1);
		puts("</HashMapSlot>");
	}
	puts("</HashMap>");
}

void HashMap2__insert(struct HashMap2 *const hm, struct HashMapData *d)
{
	int i = 0;
	size_t slot = 0;
	while (HashMapData__hash2(d->key, i) == HASH_MAP_SLOT_FULL) {
		i++;
		if (i >= hm->_current_size) {
		}
	}
}

int HashMap2__find(const struct HashMap2 *const hm, int key,
		   const struct HashMapData *d)
{
	const size_t hash_map_size = hm->_current_size;
	const struct HashMapData *const head = hm->_slotsHead;

	int i = 0;
	int h = HashMapData2__hash(key, i, hm->_current_size);
	const struct HashMapData *p;

	while (i < hash_map_size) {
		i++;
		h = HashMapData2__hash(key, i, hash_map_size);
		p = head + h;
		if (p != NULL && p->key == key) {
			d = p;
			return 0;
		}
	}
	return HASH_MAP_DATA_NOT_FOUND;
}

void HashMap2__remove(const struct HashMap2 *const hm, int key)
{
}
