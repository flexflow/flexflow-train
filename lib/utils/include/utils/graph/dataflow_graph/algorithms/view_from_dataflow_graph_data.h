#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DATAFLOW_GRAPH_ALGORITHMS_VIEW_FROM_DATAFLOW_GRAPH_DATA_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DATAFLOW_GRAPH_ALGORITHMS_VIEW_FROM_DATAFLOW_GRAPH_DATA_H

#include "utils/graph/dataflow_graph/algorithms/dataflow_graph_data.dtg.h"
#include "utils/graph/dataflow_graph/dataflow_graph_view.h"
#include "utils/graph/dataflow_graph/i_dataflow_graph_view.h"

namespace FlexFlow {

struct ViewFromDataflowGraphData final : virtual public IDataflowGraphView {

public:
  explicit ViewFromDataflowGraphData(DataflowGraphData const &);

  std::unordered_set<Node> query_nodes(NodeQuery const &query) const override;
  std::unordered_set<DataflowEdge>
      query_edges(DataflowEdgeQuery const &query) const override;
  std::unordered_set<DataflowOutput>
      query_outputs(DataflowOutputQuery const &query) const override;
  ViewFromDataflowGraphData *clone() const override;

private:
  DataflowGraphData data;
};

DataflowGraphView view_from_dataflow_graph_data(DataflowGraphData const &);

} // namespace FlexFlow

#endif
