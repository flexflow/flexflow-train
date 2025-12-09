#include "utils/graph/digraph/algorithms/get_initial_nodes.h"
#include "utils/containers/set_minus.h"
#include "utils/graph/digraph/algorithms/get_edges.h"
#include "utils/graph/node/algorithms.h"

namespace FlexFlow {

std::unordered_set<Node> get_initial_nodes(DiGraphView const &g) {
  std::unordered_set<Node> all_nodes = get_nodes(g);
  std::unordered_set<Node> with_incoming_edge =
      transform(get_edges(g), [](DirectedEdge const &e) { return e.dst; });

  return set_minus(all_nodes, with_incoming_edge);
}

} // namespace FlexFlow
