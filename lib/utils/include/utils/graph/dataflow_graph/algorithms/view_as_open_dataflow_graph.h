#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DATAFLOW_GRAPH_ALGORITHMS_VIEW_AS_OPEN_DATAFLOW_GRAPH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DATAFLOW_GRAPH_ALGORITHMS_VIEW_AS_OPEN_DATAFLOW_GRAPH_H

#include "utils/graph/dataflow_graph/dataflow_graph_view.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_graph_view.h"

namespace FlexFlow {

struct ViewDataflowGraphAsOpenDataflowGraph final
    : public IOpenDataflowGraphView {

  ViewDataflowGraphAsOpenDataflowGraph() = delete;
  ViewDataflowGraphAsOpenDataflowGraph(DataflowGraphView const &);

  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;
  std::unordered_set<DataflowOutput>
      query_outputs(DataflowOutputQuery const &) const override;
  std::unordered_set<DataflowGraphInput> get_inputs() const override;
  std::unordered_set<OpenDataflowEdge>
      query_edges(OpenDataflowEdgeQuery const &) const override;

  ViewDataflowGraphAsOpenDataflowGraph *clone() const override;

  virtual ~ViewDataflowGraphAsOpenDataflowGraph() = default;

private:
  DataflowGraphView g;
};

OpenDataflowGraphView view_as_open_dataflow_graph(DataflowGraphView const &);

} // namespace FlexFlow

#endif
