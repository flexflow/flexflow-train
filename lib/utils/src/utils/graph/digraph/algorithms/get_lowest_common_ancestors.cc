#include "utils/containers.h"
#include "utils/containers/intersection.h"
#include "utils/containers/is_subseteq_of.h"
#include "utils/containers/transform.h"
#include "utils/graph/digraph/algorithms.h"
#include "utils/graph/digraph/algorithms/get_ancestors.h"
#include "utils/graph/digraph/algorithms/get_longest_path_lengths_from_root.h"
#include "utils/graph/digraph/algorithms/is_acyclic.h"
#include "utils/graph/digraph/algorithms/is_tree.h"
#include "utils/graph/node/algorithms.h"
#include "utils/hash/unordered_set.h"
#include <optional>

namespace FlexFlow {

std::optional<std::unordered_set<Node>>
    get_lowest_common_ancestors(DiGraphView const &g,
                                std::unordered_set<Node> const &nodes) {
  assert(is_acyclic(g));
  assert(is_subseteq_of(nodes, get_nodes(g)));
  if (num_nodes(g) == 0 || nodes.size() == 0) {
    return std::nullopt;
  }
  std::unordered_set<std::unordered_set<Node>> ancestors =
      transform(nodes, [&](Node const &n) {
        return set_union(get_ancestors(g, n), {n});
      });
  std::unordered_set<Node> common_ancestors = intersection(ancestors).value();
  if (common_ancestors.empty()) {
    return common_ancestors;
  }
  std::unordered_map<Node, int> depth_levels =
      get_longest_path_lengths_from_root(g);
  int largest_depth_for_common_ancestors = maximum(transform(
      common_ancestors, [&](Node const &n) { return depth_levels.at(n); }));
  return filter(common_ancestors, [&](Node const &n) {
    return depth_levels.at(n) == largest_depth_for_common_ancestors;
  });
}

} // namespace FlexFlow
