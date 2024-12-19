#ifndef _FLEXFLOW_UTILS_GRAPH_SERIAL_PARALLEL_IS_VALID_SP_IZATION_H
#define _FLEXFLOW_UTILS_GRAPH_SERIAL_PARALLEL_IS_VALID_SP_IZATION_H

#include "utils/graph/digraph/digraph_view.h"
#include "utils/graph/series_parallel/series_parallel_decomposition.dtg.h"
#include <unordered_map>

namespace FlexFlow {

bool dependencies_are_maintained(DiGraphView const &g,
                                 SeriesParallelDecomposition const &sp);

} // namespace FlexFlow

#endif
