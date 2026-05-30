#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_VIEW_FROM_OPEN_KWARG_DATAFLOW_GRAPH_DATA_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_VIEW_FROM_OPEN_KWARG_DATAFLOW_GRAPH_DATA_H

#include "utils/graph/kwarg_dataflow_graph/kwarg_dataflow_output_query.h"
#include "utils/graph/node/node_query.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/open_kwarg_dataflow_graph_data.dtg.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/open_kwarg_dataflow_graph_data.h"
#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_edge_query.h"
#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_graph_view.h"

namespace FlexFlow {

template <typename GraphInputName, typename SlotName>
struct ViewFromOpenKwargDataflowGraphData final
    : virtual public IOpenKwargDataflowGraphView<GraphInputName, SlotName> {
  ViewFromOpenKwargDataflowGraphData(
      OpenKwargDataflowGraphData<GraphInputName, SlotName> const &data)
      : data(data) {}

  std::unordered_set<Node> query_nodes(NodeQuery const &query) const override {
    return apply_node_query(query, this->data.nodes);
  }

  std::unordered_set<KwargDataflowGraphInput<GraphInputName>>
      get_inputs() const override {
    return this->data.inputs;
  }

  std::unordered_set<OpenKwargDataflowEdge<GraphInputName, SlotName>>
      query_edges(OpenKwargDataflowEdgeQuery<GraphInputName, SlotName> const
                      &query) const override {
    return filter(
        this->data.edges,
        [&](OpenKwargDataflowEdge<GraphInputName, SlotName> const &e) {
          return open_kwarg_dataflow_edge_query_includes(query, e);
        });
  }

  std::unordered_set<KwargDataflowOutput<SlotName>> query_outputs(
      KwargDataflowOutputQuery<SlotName> const &query) const override {
    return filter(this->data.outputs,
                  [&](KwargDataflowOutput<SlotName> const &o) {
                    return kwarg_dataflow_output_query_includes(query, o);
                  });
  }

  ViewFromOpenKwargDataflowGraphData<GraphInputName, SlotName> *
      clone() const override {
    return new ViewFromOpenKwargDataflowGraphData<GraphInputName, SlotName>{
        this->data};
  }

private:
  OpenKwargDataflowGraphData<GraphInputName, SlotName> data;
};

template <typename GraphInputName, typename SlotName>
OpenKwargDataflowGraphView<GraphInputName, SlotName>
    view_from_open_kwarg_dataflow_graph_data(
        OpenKwargDataflowGraphData<GraphInputName, SlotName> const &data) {
  require_open_kwarg_dataflow_graph_data_is_valid(data);

  return OpenKwargDataflowGraphView<GraphInputName, SlotName>::template create<
      ViewFromOpenKwargDataflowGraphData<GraphInputName, SlotName>>(data);
}

} // namespace FlexFlow

#endif
