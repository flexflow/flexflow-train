#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_SERIES_REDUCTION_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_SERIES_REDUCTION_H

#include "utils/graph/multidigraph/multidiedge.dtg.h"
#include "utils/graph/multidigraph/multidigraph.h"
#include "utils/graph/series_parallel/extended_series_reduction.dtg.h"
#include "utils/graph/series_parallel/series_reduction.dtg.h"
#include "utils/hash/vector.h"

namespace FlexFlow {

Node get_pre_node(MultiDiGraphView const &, SeriesReduction const &);
Node get_post_node(MultiDiGraphView const &, SeriesReduction const &);
Node get_center_node(MultiDiGraphView const &, SeriesReduction const &);

SeriesReduction make_series_reduction(MultiDiEdge const &, MultiDiEdge const &);
std::optional<SeriesReduction> find_series_reduction(MultiDiGraphView const &);

/**
 * @brief Finds all the ExtendedSeriesReduction structures in a given graph.
 *
 * For example, in the following graph:
 *
 *     A -> B -> D -> E
 *      \            /
 *        ->  C  ->
 *
 * We have that [(A,B), (B,D), (D,E)] and [(A,C), (C,E)] both constitute
 * `ExtendedSeriesReduction`.
 */
std::unordered_set<ExtendedSeriesReduction>
    find_all_extended_series_reductions(MultiDiGraphView const &g);

MultiDiEdge apply_series_reduction(MultiDiGraph &, SeriesReduction const &);

/**
 * @brief Applies a given ExtendedSeriesReduction in-place to a given graph.
 *
 * For example, in the following graph:
 *
 *     A -> B -> D -> E
 *      \            /
 *        ->  C  ->
 *
 * Given the ExtendedSeriesReduction [(A,B), (B,D), (D,E)], the intermediate
 *nodes B, D, will be deleted, and the resulting graph will be:
 *
 *     A  ---->  E
 *      \       /
 *       -> C ->
 *
 **/
MultiDiEdge
    apply_extended_series_reduction(MultiDiGraph &g,
                                    ExtendedSeriesReduction const &reduction);

} // namespace FlexFlow

#endif
