#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLEL_GET_SERIAL_PARALLEL_DECOMPOSITION_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLEL_GET_SERIAL_PARALLEL_DECOMPOSITION_H

#include "utils/graph/digraph/digraph.h"
#include "utils/graph/serial_parallel/serial_parallel_decomposition.dtg.h"
#include "utils/optional.h"
#include <unordered_set>
#include <variant>

namespace FlexFlow {

std::optional<SerialParallelDecomposition>
    get_serial_parallel_decomposition(DiGraphView const &);
std::optional<SerialParallelDecomposition>
    get_serial_parallel_decomposition_with_dummy_nodes(
        DiGraphView const &, std::unordered_set<Node> const &);

} // namespace FlexFlow

#endif

