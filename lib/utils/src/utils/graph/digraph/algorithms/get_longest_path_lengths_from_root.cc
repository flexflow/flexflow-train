#include "utils/graph/digraph/algorithms/get_longest_path_lengths_from_root.h"
#include "utils/containers/all_of.h"
#include "utils/containers/maximum.h"
#include "utils/containers/transform.h"
#include "utils/containers/values.h"
#include "utils/graph/digraph/algorithms/get_predecessors.h"
#include "utils/graph/digraph/algorithms/get_topological_ordering.h"
#include "utils/graph/digraph/algorithms/is_acyclic.h"
#include <unordered_map>

namespace FlexFlow {

std::unordered_map<Node, float> get_weighted_longest_path_lengths_from_root(
    DiGraphView const &g, std::unordered_map<Node, float> const &node_costs) {

  assert(is_acyclic(g));
  assert(all_of(values(node_costs), [&](float cost) { return cost >= 0; }));

  std::vector<Node> topo_order = get_topological_ordering(g);
  std::unordered_map<Node, float> longest_path_lengths;

  for (Node const &n : topo_order) {
    std::unordered_set<float> predecessor_path_lengths =
        transform(get_predecessors(g, n), [&](Node const &pred) {
          return longest_path_lengths.at(pred);
        });
    longest_path_lengths[n] =
        (predecessor_path_lengths.size() == 0)
            ? node_costs.at(n)
            : maximum(predecessor_path_lengths) + node_costs.at(n);
  }
  return longest_path_lengths;
}

std::unordered_map<Node, nonnegative_int>
    get_longest_path_lengths_from_root(DiGraphView const &g) {

  assert(is_acyclic(g));

  std::vector<Node> topo_order = get_topological_ordering(g);
  std::unordered_map<Node, nonnegative_int> longest_path_lengths;

  for (Node const &n : topo_order) {
    std::unordered_set<nonnegative_int> predecessor_path_lengths =
        transform(get_predecessors(g, n), [&](Node const &pred) {
          return longest_path_lengths.at(pred);
        });
    nonnegative_int new_value = (predecessor_path_lengths.size() == 0)
                                    ? 1_n
                                    : maximum(predecessor_path_lengths) + 1_n;

    longest_path_lengths.emplace(n, new_value);
  }

  return longest_path_lengths;
}

} // namespace FlexFlow
