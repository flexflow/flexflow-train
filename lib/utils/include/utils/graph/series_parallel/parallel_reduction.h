#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_PARALLEL_REDUCTION_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_PARALLEL_REDUCTION_H

#include "utils/graph/multidigraph/multidigraph.h"
#include "utils/graph/series_parallel/parallel_reduction.dtg.h"
#include <optional>

namespace FlexFlow {

ParallelReduction make_parallel_reduction(MultiDiEdge const &,
                                          MultiDiEdge const &);
std::optional<ParallelReduction>
    find_parallel_reduction(MultiDiGraphView const &);

std::unordered_map<DirectedEdge, std::unordered_set<MultiDiEdge>>
    find_all_extended_parallel_reductions(MultiDiGraphView const &);

MultiDiEdge apply_parallel_reduction(MultiDiGraph &, ParallelReduction const &);

MultiDiEdge
    apply_extended_parallel_reduction(MultiDiGraph &,
                                      std::unordered_set<MultiDiEdge> const &);

} // namespace FlexFlow

#endif
