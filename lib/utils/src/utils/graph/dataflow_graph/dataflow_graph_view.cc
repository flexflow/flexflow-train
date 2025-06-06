#include "utils/graph/dataflow_graph/dataflow_graph_view.h"

namespace FlexFlow {

std::unordered_set<Node>
    DataflowGraphView::query_nodes(NodeQuery const &q) const {
  return this->get_interface().query_nodes(q);
}

std::unordered_set<DataflowEdge>
    DataflowGraphView::query_edges(DataflowEdgeQuery const &q) const {
  return this->get_interface().query_edges(q);
}

std::unordered_set<DataflowOutput>
    DataflowGraphView::query_outputs(DataflowOutputQuery const &q) const {
  return this->get_interface().query_outputs(q);
}

IDataflowGraphView const &DataflowGraphView::get_interface() const {
  return *std::dynamic_pointer_cast<IDataflowGraphView const>(
      GraphView::ptr.get());
}

} // namespace FlexFlow
