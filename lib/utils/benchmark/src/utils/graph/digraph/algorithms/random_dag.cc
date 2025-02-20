#include "./random_dag.h"
#include "utils/containers/vector_of.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include "utils/nonnegative_int/nonnegative_int.h"
#include "utils/random_utils.h"

namespace FlexFlow {

DiGraphView random_dag(nonnegative_int num_nodes, float edges_fraction) {
  assert(edges_fraction <= 1.0);
  assert(edges_fraction >= 0.0);

  int max_num_edges = [&] {
    int nn = num_nodes.unwrap_nonnegative();

    return (nn * (nn - 1)) / 2;
  }();

  nonnegative_int num_edges = nonnegative_int{
      static_cast<int>(max_num_edges * edges_fraction),
  };

  assert(num_edges <= max_num_edges);

  DiGraph g = DiGraph::create<AdjacencyDiGraph>();
  std::vector<Node> n = add_nodes(g, num_nodes.unwrap_nonnegative());

  std::unordered_set<DirectedEdge> edges;
  while (edges.size() < num_edges) {
    Node n1 = select_random(n);
    Node n2 = select_random(n);

    if (n1 == n2) {
      continue;
    }

    Node src = std::min(n1, n2);
    Node dst = std::max(n1, n2);

    edges.insert(DirectedEdge{src, dst});
  }

  add_edges(g, vector_of(edges));

  return g;
}

} // namespace FlexFlow
