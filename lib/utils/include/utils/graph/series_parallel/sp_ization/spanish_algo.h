#ifndef _FLEXFLOW_UTILS_GRAPH_SERIAL_PARALLEL_SP_IZATION_SPANISH_ALGO_H
#define _FLEXFLOW_UTILS_GRAPH_SERIAL_PARALLEL_SP_IZATION_SPANISH_ALGO_H

#include "utils/graph/digraph/digraph.h"
#include "utils/graph/serial_parallel/serial_parallel_decomposition.dtg.h"
#include <unordered_map>

namespace FlexFlow {

SerialParallelDecomposition one_node_at_a_time_spanish_sp_ization(DiGraph g);

} // namespace FlexFlow

#endif
