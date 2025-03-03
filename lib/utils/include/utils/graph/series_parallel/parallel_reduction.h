#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_PARALLEL_REDUCTION_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_PARALLEL_REDUCTION_H

#include "utils/graph/multidigraph/multidigraph.h"
#include "utils/graph/series_parallel/extended_parallel_reduction.dtg.h"
#include "utils/graph/series_parallel/parallel_reduction.dtg.h"
#include <optional>
#include <unordered_set>

namespace FlexFlow {

ParallelReduction make_parallel_reduction(MultiDiEdge const &,
                                          MultiDiEdge const &);

std::optional<ParallelReduction>
    find_parallel_reduction(MultiDiGraphView const &);

/**
 * @brief Finds all ExtendedParallelReduction for a given MultiDiGraph
 */
std::unordered_set<ExtendedParallelReduction>
    find_all_extended_parallel_reductions(MultiDiGraphView const &);

MultiDiEdge apply_parallel_reduction(MultiDiGraph &, ParallelReduction const &);

/**
 * @brief Applies a given ExtendedParallelReduction in place to a given
 * MultiDiGraph
 * @details The reduction removes all but one `MultiDiEdge`, so that the source,
 * destination nodes associated with the reduction become connected by a single
 * edge.
 */
MultiDiEdge
    apply_extended_parallel_reduction(MultiDiGraph &,
                                      ExtendedParallelReduction const &);

} // namespace FlexFlow

#endif
