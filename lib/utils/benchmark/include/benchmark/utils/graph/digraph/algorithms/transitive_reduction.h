#ifndef _FLEXFLOW_LIB_UTILS_BENCHMARK_INCLUDE_BENCHMARK_UTILS_DIGRAPH_ALGORITHMS_TRANSITIVE_REDUCTION_H
#define _FLEXFLOW_LIB_UTILS_BENCHMARK_INCLUDE_BENCHMARK_UTILS_DIGRAPH_ALGORITHMS_TRANSITIVE_REDUCTION_H

#include <random>

namespace FlexFlow {

void benchmark_transitive_reduction(std::mt19937 &gen,
                                    int edge_percentage,
                                    int num_nodes,
                                    bool dry_run);

}

#endif
