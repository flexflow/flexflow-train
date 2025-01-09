#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_DATAFLOW_GRAPH_ALGORITHMS_VIEW_INPUTS_AS_NODES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_DATAFLOW_GRAPH_ALGORITHMS_VIEW_INPUTS_AS_NODES_H

#include "utils/graph/dataflow_graph/dataflow_graph_view.h"
#include "utils/graph/open_dataflow_graph/dataflow_graph_input.dtg.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_graph_view.h"
#include <utility>

namespace FlexFlow {

struct ViewInputsAsNodes final : public IDataflowGraphView {

  ViewInputsAsNodes() = delete;
  ViewInputsAsNodes(OpenDataflowGraphView const &,
                    bidict<DataflowGraphInput, Node> const &);

  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;
  std::unordered_set<DataflowOutput>
      query_outputs(DataflowOutputQuery const &) const override;
  std::unordered_set<DataflowEdge>
      query_edges(DataflowEdgeQuery const &) const override;

  ViewInputsAsNodes *clone() const override;

  virtual ~ViewInputsAsNodes() = default;

private:
  OpenDataflowGraphView g;
  bidict<DataflowGraphInput, Node> input_mapping;
};

std::pair<DataflowGraphView, bidict<DataflowGraphInput, Node>>
    view_inputs_as_nodes(OpenDataflowGraphView const &);

} // namespace FlexFlow

#endif
