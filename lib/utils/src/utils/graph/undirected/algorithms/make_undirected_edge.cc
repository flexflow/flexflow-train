#include "utils/graph/undirected/algorithms/make_undirected_edge.h"
#include "utils/commutative_pair.h"

namespace FlexFlow {

UndirectedEdge make_undirected_edge(Node const &n1, Node const &n2) {
  return UndirectedEdge{commutative_pair{n1, n2}};
}

} // namespace FlexFlow
