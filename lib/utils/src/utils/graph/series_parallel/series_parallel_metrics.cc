#include "utils/graph/series_parallel/series_parallel_metrics.h"
#include "utils/containers/maximum.h"
#include "utils/containers/sum.h"
#include "utils/containers/transform.h"
#include "utils/containers/values.h"
#include "utils/containers/vector_of.h"
#include "utils/fmt/unordered_multiset.h"
#include "utils/graph/digraph/algorithms/get_edges.h"
#include "utils/graph/digraph/algorithms/get_longest_path_lengths_from_root.h"
#include "utils/graph/digraph/digraph_view.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/series_parallel/digraph_generation.h"
#include "utils/graph/series_parallel/series_parallel_decomposition.dtg.h"
#include "utils/graph/series_parallel/series_parallel_decomposition.h"
#include "utils/nonnegative_int/nonnegative_int.h"
#include "utils/variant.h"
#include <unordered_map>
namespace FlexFlow {

static std::unordered_map<Node, nonnegative_int>
    get_num_occurrences_of_nodes(Node const &node) {
  return {{node, 1_n}};
}

template <typename T>
static std::unordered_map<Node, nonnegative_int>
    get_num_occurrences_of_nodes_impl(T const &t) {
  std::unordered_map<Node, nonnegative_int> counter;
  for (Node const &node : get_nodes(t)) {
    counter.emplace(node, 0_n).first->second += 1_n;
  }
  return counter;
}

static std::unordered_map<Node, nonnegative_int>
    get_num_occurrences_of_nodes(ParallelSplit const &parallel) {
  return get_num_occurrences_of_nodes_impl(parallel);
}

static std::unordered_map<Node, nonnegative_int>
    get_num_occurrences_of_nodes(SeriesSplit const &serial) {
  return get_num_occurrences_of_nodes_impl(serial);
}

std::unordered_map<Node, nonnegative_int>
    get_num_occurrences_of_nodes(SeriesParallelDecomposition const &sp) {
  return get_num_occurrences_of_nodes_impl(sp);
}

float work_cost(SeriesParallelDecomposition const &sp,
                std::unordered_map<Node, float> cost_map) {
  return sum(transform(get_nodes(sp),
                       [&](Node const &node) { return cost_map.at(node); }));
}

float work_cost(DiGraphView const &g,
                std::unordered_map<Node, float> const &cost_map) {
  return sum(transform(vector_of(get_nodes(g)),
                       [&](Node const &node) { return cost_map.at(node); }));
}

static float critical_path_cost(Node const &node,
                                 std::unordered_map<Node, float> const &cost_map) {
  return cost_map.at(node);
}

static float critical_path_cost(SeriesSplit const &serial,
                                 std::unordered_map<Node, float> const &cost_map) {
  return sum(transform(
      serial.children, [&](std::variant<ParallelSplit, Node> const &child) {
        return critical_path_cost(widen<SeriesParallelDecomposition>(child),
                                  cost_map);
      }));
}

static float critical_path_cost(ParallelSplit const &parallel,
                                 std::unordered_map<Node, float> const &cost_map) {
  return maximum(transform(parallel.get_children(),
                           [&](std::variant<SeriesSplit, Node> const &child) {
                             return critical_path_cost(
                                 widen<SeriesParallelDecomposition>(child),
                                 cost_map);
                           }));
}

float critical_path_cost(SeriesParallelDecomposition const &sp,
                         std::unordered_map<Node, float> const &cost_map) {
  return sp.visit<float>(
      [&](auto const &t) { return critical_path_cost(t, cost_map); });
}

float critical_path_cost(DiGraphView const &g,
                         std::unordered_map<Node, float> const &cost_map) {
  return maximum(
      values(get_weighted_longest_path_lengths_from_root(g, cost_map)));
}

nonnegative_int num_dependencies(SeriesParallelDecomposition const &sp) {
  return num_dependencies(digraph_from_sp_decomposition(sp));
}

nonnegative_int num_dependencies(DiGraphView const &g) {
  return nonnegative_int{get_edges(g).size()};
}

float relative_work_increase(DiGraphView const &g,
                             SeriesParallelDecomposition const &sp,
                             std::unordered_map<Node, float> const &cost_map) {
  return work_cost(sp, cost_map) / work_cost(g, cost_map);
}

float relative_critical_path_cost_increase(
    DiGraphView const &g,
    SeriesParallelDecomposition const &sp,
    std::unordered_map<Node, float> const &cost_map) {
  return critical_path_cost(sp, cost_map) / critical_path_cost(g, cost_map);
}

float relative_num_dependencies_increase(
    DiGraphView const &g, SeriesParallelDecomposition const &sp) {
  return static_cast<float>(num_dependencies(sp).unwrap_nonnegative()) / 
         static_cast<float>(num_dependencies(g).unwrap_nonnegative());
}

} // namespace FlexFlow
