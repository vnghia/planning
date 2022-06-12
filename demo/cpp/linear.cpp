#include <iostream>

#include "dbg.h"
#include "planning/env.h"

int main() {
  Env<2, double, 3, 3> e{{3, 2}, {0.1, 0.1}, {0.3, 0.3}};
  e.Train();
  dbg(e.Q());
  return 0;
}
