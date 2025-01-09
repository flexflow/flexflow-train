#include "utils/graph/open_dataflow_graph/algorithms/view_inputs_as_nodes.h"

namespace FlexFlow {

ViewInputsAsNodes::ViewInputsAsNodes(
    OpenDataflowGraphView const &g,
    bidict<DataflowGraphInput, Node> const &input_mapping)
    : g(g), input_mapping(input_mapping) {}

std::unordered_set<Node>
    ViewInputsAsNodes::query_nodes(NodeQuery const &) const {
  NOT_IMPLEMENTED();
}

std::unordered_set<DataflowOutput>
    ViewInputsAsNodes::query_outputs(DataflowOutputQuery const &) const {
  NOT_IMPLEMENTED();
}

std::unordered_set<DataflowEdge>
    ViewInputsAsNodes::query_edges(DataflowEdgeQuery const &) const {
  NOT_IMPLEMENTED();
}

ViewInputsAsNodes *ViewInputsAsNodes::clone() const {
  return new ViewInputsAsNodes{this->g, this->input_mapping};
}

std::pair<DataflowGraphView, bidict<DataflowGraphInput, Node>>
    view_inputs_as_nodes(OpenDataflowGraphView const &) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
