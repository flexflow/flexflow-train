#include "utils/graph/series_parallel/series_parallel_metrics.h"
#include "utils/containers/maximum.h"
#include "utils/containers/sum.h"
#include "utils/containers/transform.h"
#include "utils/containers/values.h"
#include "utils/containers/vector_of.h"
#include "utils/fmt/unordered_multiset.h"
#include "utils/graph/digraph/algorithms.h"
#include "utils/graph/digraph/algorithms/get_longest_path_lengths_from_root.h"
#include "utils/graph/digraph/digraph_view.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/series_parallel/digraph_generation.h"
#include "utils/graph/series_parallel/series_parallel_decomposition.dtg.h"
#include "utils/variant.h"
#include <unordered_map>
namespace FlexFlow {

std::unordered_map<Node, size_t> get_node_frequency_map(Node const &node) {
  return {{node, 1}};
}

std::unordered_map<Node, size_t>
    get_node_frequency_map(ParallelSplit const &parallel) {
  std::unordered_map<Node, size_t> counter;
  for (std::variant<SeriesSplit, Node> const &child : parallel.get_children()) {
    for (auto const &[node, count] :
         get_node_frequency_map(widen<SeriesParallelDecomposition>(child))) {
      counter[node] += count;
    }
  }
  return counter;
}

std::unordered_map<Node, size_t>
    get_node_frequency_map(SeriesSplit const &serial) {
  std::unordered_map<Node, size_t> counter;
  for (std::variant<ParallelSplit, Node> const &child : serial.children) {
    for (auto const &[node, count] :
         get_node_frequency_map(widen<SeriesParallelDecomposition>(child))) {
      counter[node] += count;
    }
  }
  return counter;
}

std::unordered_map<Node, size_t>
    get_node_frequency_map(SeriesParallelDecomposition const &sp) {
  return sp.visit<std::unordered_map<Node, size_t>>(
      [](auto const &t) { return get_node_frequency_map(t); });
}

float work_cost(SeriesParallelDecomposition const &sp,
                std::unordered_map<Node, float> cost_map) {
  auto cost_per_node_group = [&](std::pair<Node, float> const &pair) {
    return pair.second * cost_map.at(pair.first);
  };
  std::unordered_map<Node, size_t> counter = get_node_frequency_map(sp);
  std::vector<std::pair<Node, size_t>> pairs(counter.cbegin(), counter.cend());
  return sum(transform(pairs, cost_per_node_group));
}

float work_cost(DiGraphView const &g,
                std::unordered_map<Node, float> const &cost_map) {
  return sum(transform(vector_of(get_nodes(g)),
                       [&](Node const &node) { return cost_map.at(node); }));
}

float critical_path_cost(Node const &node,
                         std::unordered_map<Node, float> const &cost_map) {
  return cost_map.at(node);
}

float critical_path_cost(SeriesSplit const &serial,
                         std::unordered_map<Node, float> const &cost_map) {
  return sum(transform(
      serial.children, [&](std::variant<ParallelSplit, Node> const &child) {
        return critical_path_cost(widen<SeriesParallelDecomposition>(child),
                                  cost_map);
      }));
}

float critical_path_cost(ParallelSplit const &parallel,
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

int num_dependencies(SeriesParallelDecomposition const &sp) {
  return num_dependencies(digraph_from_sp_decomposition(sp));
}

int num_dependencies(DiGraphView const &g) {
  return num_edges(g);
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
  return static_cast<float>(num_dependencies(sp)) / num_dependencies(g);
}

} // namespace FlexFlow
