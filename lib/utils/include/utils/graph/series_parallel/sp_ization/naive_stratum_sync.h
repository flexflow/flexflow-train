#ifndef _FLEXFLOW_UTILS_GRAPH_SERIES_PARALLEL_NAIVE_STRATUM_SYNC_H
#define _FLEXFLOW_UTILS_GRAPH_SERIES_PARALLEL_NAIVE_STRATUM_SYNC_H

#include "utils/graph/digraph/digraph_view.h"
#include "utils/graph/series_parallel/series_parallel_decomposition.dtg.h"

namespace FlexFlow {

/**
 * \brief See \ref spization-naive.
 **/
SeriesParallelDecomposition naive_stratum_sync_sp_ization(DiGraphView const &g);

} // namespace FlexFlow

#endif
