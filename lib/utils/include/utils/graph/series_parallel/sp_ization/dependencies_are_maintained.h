#ifndef _FLEXFLOW_UTILS_GRAPH_SERIAL_PARALLEL_IS_VALID_SP_IZATION_H
#define _FLEXFLOW_UTILS_GRAPH_SERIAL_PARALLEL_IS_VALID_SP_IZATION_H

#include "utils/graph/digraph/digraph_view.h"
#include "utils/graph/series_parallel/series_parallel_decomposition.dtg.h"
#include <unordered_map>

namespace FlexFlow {
/**
 * @brief Checks if dependencies are maintained between a directed graph and its
 * series-parallel decomposition.
 *
 * @details This function ensures that the series-parallel decomposition is a
 * valid sp-ization of the given directed graph, by checking that dependencies
 * are maintained. Dependencies are considered maintained if:
 * - Both the directed graph and the series-parallel decomposition contain the
 * same set of nodes.
 * - For every node in the directed graph, all its ancestors are also ancestors
 * within the series-parallel decomposition.
 *
 */
bool dependencies_are_maintained(DiGraphView const &g,
                                 SeriesParallelDecomposition const &sp);

} // namespace FlexFlow

#endif
