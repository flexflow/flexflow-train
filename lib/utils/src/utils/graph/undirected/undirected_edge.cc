#include "utils/graph/undirected/undirected_edge.h"
#include "utils/hash/tuple.h"
#include <sstream>

namespace FlexFlow {

bool is_connected_to(UndirectedEdge const &e, Node const &n) {
  return e.endpoints.min() == n || e.endpoints.max() == n;
}

std::unordered_set<Node> get_endpoints(UndirectedEdge const &e) {
  return {e.endpoints.min(), e.endpoints.max()};
}

} // namespace FlexFlow
