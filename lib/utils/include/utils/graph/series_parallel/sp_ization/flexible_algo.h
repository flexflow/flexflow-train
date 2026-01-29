#ifndef _FLEXFLOW_UTILS_GRAPH_SERIAL_PARALLEL_SP_IZATION_FLEXIBLE_ALGO_H
#define _FLEXFLOW_UTILS_GRAPH_SERIAL_PARALLEL_SP_IZATION_FLEXIBLE_ALGO_H

#include "utils/graph/digraph/digraph.h"
#include "utils/graph/digraph/digraph_view.h"
#include "utils/graph/digraph/directed_edge.dtg.h"
#include "utils/graph/series_parallel/series_parallel_decomposition.dtg.h"
#include "utils/graph/series_parallel/sp_ization/node_role.dtg.h"
#include "utils/graph/series_parallel/sp_ization/up_down_partition.dtg.h"
#include <unordered_map>
#include <unordered_set>

namespace FlexFlow {

SeriesParallelDecomposition
    flexible_sync(DiGraphView const &g,
                  std::unordered_map<Node, float> const &cost_map);

} // namespace FlexFlow

#endif
