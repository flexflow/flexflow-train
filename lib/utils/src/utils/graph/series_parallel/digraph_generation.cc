#include "utils/graph/series_parallel/digraph_generation.h"
#include "utils/containers/transform.h"
#include "utils/containers/vector_of.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/digraph/algorithms.h"
#include "utils/graph/digraph/algorithms/materialize_digraph_view.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/series_parallel/series_parallel_splits.h"
#include "utils/variant.h"

namespace FlexFlow {

std::unordered_map<Node, Node> parallel_extend(DiGraph &g,
                                               DiGraphView const &ext) {
  std::unordered_map<Node, Node> node_map;
  for (Node const &node : get_nodes(ext)) {
    node_map.emplace(node, g.add_node());
  }
  for (DirectedEdge const &edge : get_edges(ext)) {
    g.add_edge(DirectedEdge{node_map.at(edge.src), node_map.at(edge.dst)});
  }
  return node_map;
}

std::unordered_map<Node, Node> serial_extend(DiGraph &g,
                                             DiGraphView const &ext) {
  std::unordered_set<Node> original_sinks = get_sinks(g);
  std::unordered_map<Node, Node> node_map = parallel_extend(g, ext);
  for (Node const &node1 : original_sinks) {
    for (Node const &node2 : get_sources(ext)) {
      g.add_edge(DirectedEdge{node1, node_map.at(node2)});
    }
  }
  return node_map;
}

DiGraph series_composition(DiGraphView const &g1, DiGraphView const &g2) {
  DiGraph g = materialize_digraph_view<AdjacencyDiGraph>(g1);
  serial_extend(g, g2);
  return g;
}

DiGraph parallel_composition(DiGraphView const &g1, DiGraphView const &g2) {
  DiGraph g = materialize_digraph_view<AdjacencyDiGraph>(g1);
  parallel_extend(g, g2);
  return g;
}

DiGraph series_composition(std::vector<DiGraphView> const &graphs) {
  DiGraph g = DiGraph::create<AdjacencyDiGraph>();
  for (DiGraphView const &gs : graphs) {
    g = materialize_digraph_view<AdjacencyDiGraph>(series_composition(g, gs));
  }
  return g;
}

// TODO(@pietro): should be std::unordered_set<DiGraphView>, but DiGraphs are
// currently non-hashable
DiGraph parallel_composition(std::vector<DiGraphView> const &graphs) {
  DiGraph g = DiGraph::create<AdjacencyDiGraph>();
  for (DiGraphView const &gs : graphs) {
    g = materialize_digraph_view<AdjacencyDiGraph>(parallel_composition(g, gs));
  }
  return g;
}

DiGraph digraph_from_sp_decomposition(Node const &node) {
  DiGraph g = DiGraph::create<AdjacencyDiGraph>();
  g.add_node();
  return g;
}

DiGraph digraph_from_sp_decomposition(SeriesSplit const &serial) {
  std::vector<SeriesParallelDecomposition> children =
      transform(serial.children, [](auto const &child) {
        return widen<SeriesParallelDecomposition>(child);
      });
  return series_composition(
      transform(children, [](auto const child) -> DiGraphView {
        return digraph_from_sp_decomposition(child);
      }));
}

DiGraph digraph_from_sp_decomposition(ParallelSplit const &parallel) {
  std::vector<SeriesParallelDecomposition> children =
      transform(vector_of(parallel.get_children()), [](auto const &child) {
        return widen<SeriesParallelDecomposition>(child);
      });
  return parallel_composition(
      transform(children, [](auto const child) -> DiGraphView {
        return digraph_from_sp_decomposition(child);
      }));
}

DiGraph digraph_from_sp_decomposition(SeriesParallelDecomposition const &sp) {
  return sp.visit<DiGraph>(
      [](auto const &x) { return digraph_from_sp_decomposition(x); });
}

} // namespace FlexFlow
