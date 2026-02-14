#include "utils/graph/digraph/algorithms/get_descendants.h"
#include "utils/containers/contains.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/digraph/algorithms/get_successors.h"
#include "utils/graph/digraph/algorithms/is_acyclic.h"
#include "utils/graph/digraph/digraph_view.h"
#include "utils/graph/node/algorithms.h"

namespace FlexFlow {
std::unordered_set<Node> get_descendants(DiGraphView const &g,
                                         Node const &starting_node) {
  assert(is_acyclic(g));
  assert(contains(get_nodes(g), starting_node));

  return unordered_set_of(get_bfs_ordering(g, get_successors(g, starting_node)));
};

} // namespace FlexFlow
