#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLLEL_GET_ANCESTORS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLLEL_GET_ANCESTORS_H

#include "utils/graph/series_parallel/series_parallel_decomposition.dtg.h"

namespace FlexFlow {

std::unordered_set<Node> get_ancestors(SeriesParallelDecomposition const &sp,
                                       Node const &starting_node);

} // namespace FlexFlow

#endif
