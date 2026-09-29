#include "utils/graph/digraph/algorithms/transitive_reduction.h"
#include "internal/random_dag.h"
#include "utils/benchmark_utils.h"

namespace FlexFlow {

void benchmark_transitive_reduction(std::mt19937 &gen,
                                    int edge_percentage,
                                    int num_nodes,
                                    bool dry_run) {

  DiGraphView g = random_dag(gen,
                             nonnegative_int{num_nodes},
                             static_cast<float>(edge_percentage) / 100.0);

  LOOP(1000, dry_run) {
    transitive_reduction(g);
  }
}

} // namespace FlexFlow
