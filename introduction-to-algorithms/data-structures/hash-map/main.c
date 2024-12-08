#include <stdio.h>

#include "hash-map.h"

int main() {
  struct HashMapData *d = HashMapData(1, 1);
  struct HashMapData *d2 = HashMapData(17, 2);
  struct HashMapData *d3 = HashMapData(81, 3);
  struct HashMapData *d4 = HashMapData(53, 4);
  struct HashMap *hm = HashMap();
  HashMap__insert(hm, d);
  HashMap__insert(hm, d2);
  HashMap__insert(hm, d3);
  HashMap__insert(hm, d4);
  HashMap__show(hm);

  const int search_key = 17;
  struct HashMapData *d5;
  int found = HashMap__find(hm, search_key, d5);
  if (found != HASH_MAP_DATA_NOT_FOUND) {
    printf("Found key=%d at loc=%p\n", search_key, d5);
    HashMapData__show(d5);
  } else {
    printf("Could not find item with key=%d\n", 17);
  }

  HashMap__del(hm);
  return 0;
}
