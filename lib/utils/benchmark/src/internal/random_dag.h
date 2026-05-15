#ifndef _FLEXFLOW_LIB_UTILS_BENCHMARK_SRC_UTILS_GRAPH_DIGRAPH_ALGORITHMS_RANDOM_DAG_H
#define _FLEXFLOW_LIB_UTILS_BENCHMARK_SRC_UTILS_GRAPH_DIGRAPH_ALGORITHMS_RANDOM_DAG_H

#include "utils/graph/digraph/digraph_view.h"
#include "utils/nonnegative_int/nonnegative_int.h"

namespace FlexFlow {

DiGraphView random_dag(nonnegative_int num_nodes, float edges_fraction);

} // namespace FlexFlow

#endif
