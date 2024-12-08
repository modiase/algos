#include "hash-map.h"

int main() {
  struct HashMap *m = HashMap();
  HashMap__show(m);
  HashMap__del(m);
  return 0;
}
