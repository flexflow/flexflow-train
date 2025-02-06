#include "utils/graph/dataflow_graph/algorithms/view_as_open_dataflow_graph.h"
#include "utils/containers/transform.h"

namespace FlexFlow {

ViewDataflowGraphAsOpenDataflowGraph::ViewDataflowGraphAsOpenDataflowGraph(
    DataflowGraphView const &g)
    : g(g) {}

std::unordered_set<Node> ViewDataflowGraphAsOpenDataflowGraph::query_nodes(
    NodeQuery const &q) const {
  return this->g.query_nodes(q);
}

std::unordered_set<OpenDataflowEdge>
    ViewDataflowGraphAsOpenDataflowGraph::query_edges(
        OpenDataflowEdgeQuery const &q) const {
  std::unordered_set<DataflowEdge> closed_edges =
      this->g.query_edges(q.standard_edge_query);

  return transform(closed_edges,
                   [](DataflowEdge const &e) { return OpenDataflowEdge{e}; });
}

std::unordered_set<DataflowOutput>
    ViewDataflowGraphAsOpenDataflowGraph::query_outputs(
        DataflowOutputQuery const &q) const {
  return this->g.query_outputs(q);
}

std::unordered_set<DataflowGraphInput>
    ViewDataflowGraphAsOpenDataflowGraph::get_inputs() const {
  return {};
}

ViewDataflowGraphAsOpenDataflowGraph *
    ViewDataflowGraphAsOpenDataflowGraph::clone() const {
  return new ViewDataflowGraphAsOpenDataflowGraph{this->g};
}

OpenDataflowGraphView view_as_open_dataflow_graph(DataflowGraphView const &g) {
  return OpenDataflowGraphView::create<ViewDataflowGraphAsOpenDataflowGraph>(g);
}

} // namespace FlexFlow
