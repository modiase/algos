#include <stdlib.h>

#include "hash-map.h"

int main() {

  struct HashMap *hm = HashMap();

  const size_t ndata = 10;
  struct HashMapData *data =
      (struct HashMapData *)malloc(ndata * sizeof(struct HashMapData));
  for (size_t i = 0; i < ndata; i++) {
    struct HashMapData *d = data + i;
    d->data = i;
    d->key = i;
    HashMap__insert(hm, d);
  }

  HashMap__show(hm);

  HashMap__del(hm);
  return 0;
}
