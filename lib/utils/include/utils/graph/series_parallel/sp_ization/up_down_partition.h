#ifndef _FLEXFLOW_UTILS_GRAPH_SERIES_PARALLEL_SP_IZATION_UP_DOWN_PARTITION_H
#define _FLEXFLOW_UTILS_GRAPH_SERIES_PARALLEL_SP_IZATION_UP_DOWN_PARTITION_H

#include "utils/graph/digraph/digraph.h"
#include "utils/graph/series_parallel/sp_ization/up_down_partition.dtg.h"
#include <unordered_set>

namespace FlexFlow {

std::unordered_set<Node> get_up_frontier(DiGraph const &sp,
                                         UpDownPartition const &partition);

std::unordered_set<Node> get_down_frontier(DiGraph const &sp,
                                           UpDownPartition const &partition);

} // namespace FlexFlow

#endif

