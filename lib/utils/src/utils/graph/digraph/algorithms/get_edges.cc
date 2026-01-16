#include "utils/graph/digraph/algorithms/get_edges.h"
#include "utils/graph/digraph/directed_edge_query.h"

namespace FlexFlow {

size_t num_edges(DiGraphView const &g) {
  return get_edges(g).size();
}

std::unordered_set<DirectedEdge> get_edges(DiGraphView const &g) {
  return g.query_edges(directed_edge_query_all());
}

} // namespace FlexFlow
