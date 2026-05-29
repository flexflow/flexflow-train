#include "utils/graph/multidigraph/algorithms/get_multidiedge_to_diedge_map.h"
#include "utils/containers/generate_unordered_map.h"
#include "utils/graph/multidigraph/algorithms/get_directed_edge.h"
#include "utils/graph/multidigraph/algorithms/get_edges.h"

namespace FlexFlow {

std::unordered_map<MultiDiEdge, DirectedEdge>
    get_multidiedge_to_diedge_map(MultiDiGraphView const &g) {
  return generate_unordered_map(get_edges(g), [&](MultiDiEdge const &e) {
    return get_directed_edge(g, e);
  });
}

} // namespace FlexFlow
