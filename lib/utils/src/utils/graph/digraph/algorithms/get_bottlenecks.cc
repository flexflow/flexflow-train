#include "utils/graph/digraph/algorithms/get_bottlenecks.h"
#include "utils/containers/filter.h"
#include "utils/containers/get_only.h"
#include "utils/containers/set_difference.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/digraph/algorithms/get_initial_nodes.h"
#include "utils/graph/digraph/algorithms/get_terminal_nodes.h"
#include "utils/graph/digraph/algorithms/get_weakly_connected_components.h"
#include "utils/graph/digraph/algorithms/is_acyclic.h"
#include "utils/graph/node/algorithms.h"

namespace FlexFlow {
std::unordered_set<Node> get_bottlenecks(DiGraphView const &g) {
  ASSERT(is_acyclic(g));
  ASSERT(get_weakly_connected_components(g).size() ==
         1); // must be singly connected

  std::unordered_set<Node> bottlenecks = filter(get_nodes(g), [&](Node const &n) {
    DiGraphView subgraph = get_subgraph(g, set_difference(get_nodes(g), {n}));
    return get_weakly_connected_components(subgraph).size() == 2;
  });

  if (get_initial_nodes(g).size() == 1) {
    bottlenecks.insert(get_only(get_initial_nodes(g)));
  }

  if (get_terminal_nodes(g).size() == 1) {
    bottlenecks.insert(get_only(get_terminal_nodes(g)));
  }

  return bottlenecks;
}

} // namespace FlexFlow
