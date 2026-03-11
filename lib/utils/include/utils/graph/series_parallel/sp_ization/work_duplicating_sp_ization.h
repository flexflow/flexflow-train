#ifndef _FLEXFLOW_UTILS_GRAPH_SERIES_PARALLEL_WORK_DUPLICATING_SP_IZATION_H
#define _FLEXFLOW_UTILS_GRAPH_SERIES_PARALLEL_WORK_DUPLICATING_SP_IZATION_H

#include "utils/graph/digraph/digraph_view.h"
#include "utils/graph/series_parallel/series_parallel_decomposition.dtg.h"
#include <unordered_map>

namespace FlexFlow {

/**
 * \brief See \ref spization-naive-work-duplicating.
 */
SeriesParallelDecomposition
    naive_work_duplicating_sp_ization(DiGraphView const &g);

/**
 * @brief See \ref spization-with-coalescing.
 */
SeriesParallelDecomposition
    work_duplicating_sp_ization_with_coalescing(DiGraphView const &g);

} // namespace FlexFlow

#endif
