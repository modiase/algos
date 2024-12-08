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
  HashMap__del(hm);
  return 0;
}
