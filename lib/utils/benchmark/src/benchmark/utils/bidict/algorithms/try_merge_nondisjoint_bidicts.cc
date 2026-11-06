#include "benchmark/utils/bidict/algorithms/try_merge_nondisjoint_bidicts.h"
#include "utils/benchmark_utils/loop.h"
#include "utils/benchmark_utils/random_bidict.h"
#include "utils/benchmark_utils/random_hex_string.h"
#include "utils/benchmark_utils/random_int.h"
#include "utils/bidict/algorithms/try_merge_nondisjoint_bidicts.h"

namespace FlexFlow {

void benchmark_try_merge_nondisjoint_bidicts(bool dry_run) {
  std::mt19937 gen;
  gen.seed(0);

  auto random_l = [](std::mt19937 &gen) -> std::string {
    return random_hex_string(gen, 10_n);
  };

  auto random_r = [](std::mt19937 &gen) -> int { return random_int(gen); };

  bidict<std::string, int> lhs = random_bidict(gen, 50_n, random_l, random_r);
  bidict<std::string, int> rhs = random_bidict(gen, 50_n, random_l, random_r);

  LOOP(100, dry_run) {
    try_merge_nondisjoint_bidicts(lhs, rhs);
  }
}

} // namespace FlexFlow
