#include "utils/graph/digraph/algorithms/get_dominators.h"
#include "utils/containers/restrict_keys.h"
#include "utils/containers/values.h"
#include "utils/graph/digraph/algorithms/get_dominators_map.h"
#include "utils/hash/unordered_set.h"
#include <queue>
#include "utils/containers/set_intersection.h"
#include "utils/optional.h"

namespace FlexFlow {

std::unordered_set<Node> get_dominators(DiGraphView const &g, Node const &n) {
  return get_dominators_map(g).at(n);
}

std::unordered_set<Node> get_dominators(DiGraphView const &g,
                                        std::unordered_set<Node> const &n) {
  ASSERT(n.size() > 0, "Cannot find dominators of no nodes");

  std::optional<std::unordered_set<Node>> result =
      set_intersection(values(restrict_keys(get_dominators_map(g), n)));

  return assert_unwrap(result);
}

} // namespace FlexFlow
