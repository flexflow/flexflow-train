#include "utils/containers/contains.h"
#include "utils/benchmark_utils.h"
#include <random>

namespace FlexFlow {

void benchmark_contains_for_set(std::mt19937 &gen, bool dry_run) {
  std::set<int> s;
  std::uniform_int_distribution<> dist(1, 50000);
  for (int i = 0; i < 1000; i++) {
    s.insert(dist(gen));
  }

  LOOP(10000, dry_run) {
    contains(s, dist(gen));
  }
}

} // namespace FlexFlow
