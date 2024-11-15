#ifndef _FLEXFLOW_UTILS_GRAPH_SERIAL_PARALLEL_IS_VALID_SP_IZATION_H
#define _FLEXFLOW_UTILS_GRAPH_SERIAL_PARALLEL_IS_VALID_SP_IZATION_H

#include "utils/graph/digraph/digraph_view.h"
#include "utils/graph/serial_parallel/serial_parallel_decomposition.dtg.h"
#include <unordered_map>

namespace FlexFlow {

bool is_valid_sp_ization(DiGraphView const &g,
                         SerialParallelDecomposition const &sp);

} // namespace FlexFlow

#endif
