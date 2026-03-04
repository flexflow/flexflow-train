#ifndef _FLEXFLOW_UTILS_GRAPH_SERIES_PARALLEL_SP_IZATION_UP_DOWN_PARTITION_H
#define _FLEXFLOW_UTILS_GRAPH_SERIES_PARALLEL_SP_IZATION_UP_DOWN_PARTITION_H

#include "utils/graph/digraph/digraph.h"
#include "utils/graph/series_parallel/sp_ization/up_down_partition.dtg.h"
#include <unordered_set>

namespace FlexFlow {

/**
 * @brief Returns the nodes n in the up set such that in the up subgraph, there
 * is no outgoing edge from n.
 */
std::unordered_set<Node> get_up_frontier(DiGraph const &sp,
                                         UpDownPartition const &partition);

/**
 * @brief Returns the nodes n in the down set such that in the down subgraph,
 * there is no incoming edge to n.
 */
std::unordered_set<Node> get_down_frontier(DiGraph const &sp,
                                           UpDownPartition const &partition);

} // namespace FlexFlow

#endif
