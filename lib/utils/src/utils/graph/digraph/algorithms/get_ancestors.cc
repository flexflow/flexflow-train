#include "utils/graph/digraph/algorithms/get_ancestors.h"
#include "utils/graph/digraph/algorithms/flipped.h"
#include "utils/graph/digraph/algorithms/get_descendants.h"
#include "utils/graph/digraph/algorithms/is_acyclic.h"

namespace FlexFlow {
std::unordered_set<Node> get_ancestors(DiGraphView const &g,
                                       Node const &starting_node) {
  assert(is_acyclic(g));
  return get_descendants(flipped(g), starting_node);
}
} // namespace FlexFlow
