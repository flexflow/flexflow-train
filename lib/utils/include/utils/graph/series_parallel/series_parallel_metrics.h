#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_METRICS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_METRICS_H

#include "utils/graph/digraph/digraph_view.h"
#include "utils/graph/series_parallel/series_parallel_decomposition.dtg.h"
#include "utils/nonnegative_int/nonnegative_int.h"
#include <unordered_map>

namespace FlexFlow {

/**
 * @brief Maps each node to the number of times it appears in the decomposition.
 *
 */
std::unordered_map<Node, nonnegative_int> get_node_counter_map(Node const &node);
std::unordered_map<Node, nonnegative_int>
    get_node_counter_map(SeriesSplit const &serial);
std::unordered_map<Node, nonnegative_int>
    get_node_counter_map(ParallelSplit const &parallel);
std::unordered_map<Node, nonnegative_int>
    get_node_counter_map(SeriesParallelDecomposition const &sp);

/**
 * @brief Calculates the total cumulative cost of all nodes in the
 * decomposition.
 *
 */
float work_cost(SeriesParallelDecomposition const &sp,
                std::unordered_map<Node, float> cost_map);

float work_cost(DiGraphView const &g,
                std::unordered_map<Node, float> const &cost_map);

/**
 * @brief Computes the total number of edges the decomposition has when viewed
 * as a DiGraph where Series connections are all to all.
 *
 */
int num_dependencies(SeriesParallelDecomposition const &sp);

int num_dependencies(DiGraphView const &g);

float critical_path_cost(SeriesParallelDecomposition const &sp,
                         std::unordered_map<Node, float> const &cost_map);

float critical_path_cost(DiGraphView const &g,
                         std::unordered_map<Node, float> const &cost_map);

/**
 * @brief Calculates the relative increase in total work cost between  the
 * original (possibly non-series-parallel) graph and a possible series-parallel
 * decomposition of that graph.
 */
float relative_work_increase(DiGraphView const &g,
                             SeriesParallelDecomposition const &sp,
                             std::unordered_map<Node, float> const &cost_map);

/**
 * @brief Calculates the relative increase in critical path cost between the
 * original (possibly non-series-parallel) graph and a possible series-parallel
 * decomposition of that graph.
 */
float relative_critical_path_cost_increase(
    DiGraphView const &g,
    SeriesParallelDecomposition const &sp,
    std::unordered_map<Node, float> const &cost_map);

/**
 * @brief Calculates the relative increase in the number of dependencies between
 * the original (possibly non-series-parallel) graph and a possible
 * series-parallel decomposition of that graph.
 */
float relative_num_dependencies_increase(DiGraphView const &g,
                                         SeriesParallelDecomposition const &sp);

} // namespace FlexFlow

#endif // _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_METRICS_H
