#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_GET_SERIES_PARALLEL_DECOMPOSITION_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_GET_SERIES_PARALLEL_DECOMPOSITION_H

#include "utils/graph/digraph/digraph.h"
#include "utils/graph/series_parallel/series_parallel_decomposition.dtg.h"
#include "utils/optional.h"

namespace FlexFlow {

std::optional<SeriesParallelDecomposition>
    get_series_parallel_decomposition(DiGraphView const &);

/**
 * @brief Unoptimized version of get_series_parallel_decomposition, used for
 * reference.
 */
std::optional<SeriesParallelDecomposition>
    get_series_parallel_decomposition_unoptimized(DiGraphView const &g);

} // namespace FlexFlow

#endif
