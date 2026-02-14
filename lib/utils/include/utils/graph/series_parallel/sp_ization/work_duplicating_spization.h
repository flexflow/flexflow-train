#ifndef _FLEXFLOW_UTILS_GRAPH_SERIES_PARALLEL_WORK_DUPLICATING_SPIZATION_H
#define _FLEXFLOW_UTILS_GRAPH_SERIES_PARALLEL_WORK_DUPLICATING_SPIZATION_H

#include "utils/graph/digraph/digraph_view.h"
#include "utils/graph/series_parallel/series_parallel_decomposition.dtg.h"
#include <unordered_map>

namespace FlexFlow {

/**
 * @brief See @ref lib/utils/include/utils/graph/series_parallel/sp_ization/README.md "README.md" for explanation.
 */
SeriesParallelDecomposition naive_work_duplicating_spization(DiGraphView const &g);

/**
 * @brief See @ref lib/utils/include/utils/graph/series_parallel/sp_ization/README.md "README.md" for explanation.
 */
SeriesParallelDecomposition
    work_duplicating_spization_with_coalescing(DiGraphView const &g);

} // namespace FlexFlow

#endif
