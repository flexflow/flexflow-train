#include "utils/graph/digraph/algorithms/is_tree.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/digraph/algorithms.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/undirected/algorithms/get_connected_components.h"

namespace FlexFlow {

bool is_tree(DiGraphView const &g) {
  assert(num_nodes(g) > 0);

  bool has_single_root = get_sources(g).size() == 1;
  bool is_connected = get_connected_components(as_undirected(g)).size() == 1;
  return has_single_root && is_connected && (num_nodes(g) - num_edges(g) == 1);
}

} // namespace FlexFlow
