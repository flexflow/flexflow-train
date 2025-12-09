#include "utils/graph/digraph/algorithms/get_terminal_nodes.h"
#include "utils/graph/digraph/algorithms/get_initial_nodes.h"
#include "utils/graph/digraph/algorithms/flipped.h"

namespace FlexFlow {

std::unordered_set<Node> get_terminal_nodes(DiGraphView const &g) {
  return get_initial_nodes(flipped(g));
}

} // namespace FlexFlow
