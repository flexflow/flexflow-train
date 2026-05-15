#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_ALGORITHMS_VIEW_FROM_KWARG_DATAFLOW_GRAPH_DATA_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_ALGORITHMS_VIEW_FROM_KWARG_DATAFLOW_GRAPH_DATA_H

#include "utils/graph/kwarg_dataflow_graph/algorithms/kwarg_dataflow_graph_data.dtg.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/kwarg_dataflow_graph_data.h"
#include "utils/graph/kwarg_dataflow_graph/i_kwarg_dataflow_graph_view.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/view_from_open_kwarg_dataflow_graph_data.h"

namespace FlexFlow {

template <typename SlotName>
struct ViewFromKwargDataflowGraphData final
    : virtual public IKwargDataflowGraphView<SlotName> {
  explicit ViewFromKwargDataflowGraphData(
      KwargDataflowGraphData<SlotName> const &data)
      : data(data) {}

  std::unordered_set<Node> query_nodes(NodeQuery const &query) const override {
    return apply_node_query(query, this->data.nodes);
  }

  std::unordered_set<KwargDataflowEdge<SlotName>> query_edges(
      KwargDataflowEdgeQuery<SlotName> const &query) const override {
    return filter(this->data.edges, [&](KwargDataflowEdge<SlotName> const &e) {
      return kwarg_dataflow_edge_query_includes(query, e);
    });
  }

  std::unordered_set<KwargDataflowOutput<SlotName>> query_outputs(
      KwargDataflowOutputQuery<SlotName> const &query) const override {
    return filter(this->data.outputs,
                  [&](KwargDataflowOutput<SlotName> const &o) {
                    return kwarg_dataflow_output_query_includes(query, o);
                  });
  }

  ViewFromKwargDataflowGraphData<SlotName> *clone() const override {
    return new ViewFromKwargDataflowGraphData<SlotName>{this->data};
  }

private:
  KwargDataflowGraphData<SlotName> data;
};

template <typename SlotName>
KwargDataflowGraphView<SlotName> view_from_kwarg_dataflow_graph_data(
    KwargDataflowGraphData<SlotName> const &data) {
  require_kwarg_dataflow_graph_data_is_valid(data);

  return KwargDataflowGraphView<SlotName>::template create<
      ViewFromKwargDataflowGraphData<SlotName>>(data);
}

} // namespace FlexFlow

#endif
