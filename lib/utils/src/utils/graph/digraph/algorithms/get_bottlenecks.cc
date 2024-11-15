#include "utils/graph/digraph/algorithms/get_bottlenecks.h"
#include "utils/containers/get_only.h"
#include "utils/containers/set_difference.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/digraph/algorithms/get_weakly_connected_components.h"
#include "utils/graph/digraph/algorithms/is_acyclic.h"
#include "utils/graph/node/algorithms.h"

namespace FlexFlow {
std::unordered_set<Node> get_bottlenecks(DiGraphView const &g) {
  std::unordered_set<Node> bottlenecks;
  assert(is_acyclic(g));
  assert(get_weakly_connected_components(g).size() ==
         1); // must be singly connected

  for (Node const &n : get_nodes(g)) {
    DiGraphView subgraph = get_subgraph(g, set_difference(get_nodes(g), {n}));
    if (get_weakly_connected_components(subgraph).size() == 2) {
      bottlenecks.insert(n);
    }
  }

  if (get_sources(g).size() == 1) {
    bottlenecks.insert(get_only(get_sources(g)));
  }

  if (get_sinks(g).size() == 1) {
    bottlenecks.insert(get_only(get_sinks(g)));
  }

  return bottlenecks;
}

} // namespace FlexFlow
