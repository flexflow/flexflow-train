#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_VIEW_FROM_OPEN_KWARG_DATAFLOW_GRAPH_DATA_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_VIEW_FROM_OPEN_KWARG_DATAFLOW_GRAPH_DATA_H

#include "utils/graph/open_kwarg_dataflow_graph/algorithms/open_kwarg_dataflow_graph_data.dtg.h"
#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_graph_view.h"
#include "utils/graph/node/node_query.h"
#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_edge_query.h"
#include "utils/graph/kwarg_dataflow_graph/kwarg_dataflow_output_query.h"

namespace FlexFlow {

template <typename GraphInputName, typename SlotName>
struct ViewFromOpenKwargDataflowGraphView final
    : virtual public IOpenKwargDataflowGraphView<GraphInputName, SlotName>
{
  ViewFromOpenKwargDataflowGraphView(OpenKwargDataflowGraphData<GraphInputName, SlotName> const &data)
    : data(data) { }

  std::unordered_set<Node> query_nodes(NodeQuery const &query) const override {
    return apply_node_query(query, this->data.nodes);
  }

  std::unordered_set<KwargDataflowGraphInput<GraphInputName>> get_inputs() const override {
    return this->data.inputs;
  }

  std::unordered_set<OpenKwargDataflowEdge<GraphInputName, SlotName>>
      query_edges(OpenKwargDataflowEdgeQuery<GraphInputName, SlotName> const &query) const override {
    return filter(this->data.edges, 
                  [&](OpenKwargDataflowEdge<GraphInputName, SlotName> const &e) {
                    return open_kwarg_dataflow_edge_query_includes(query, e);
                  });
  }

  std::unordered_set<KwargDataflowOutput<SlotName>>
      query_outputs(KwargDataflowOutputQuery<SlotName> const &query) const override {
    return filter(this->data.outputs,
                  [&](KwargDataflowOutput<SlotName> const &o) {
                    return kwarg_dataflow_output_query_includes(query, o);
                  });
  }

  ViewFromOpenKwargDataflowGraphView<GraphInputName, SlotName> *clone() const override {
    return new ViewFromOpenKwargDataflowGraphView<GraphInputName, SlotName>{this->data};
  }
private:
  OpenKwargDataflowGraphData<GraphInputName, SlotName> data;
};

template <typename GraphInputName, typename SlotName>
OpenKwargDataflowGraphView<GraphInputName, SlotName>
  view_from_open_kwarg_dataflow_graph_data(
    OpenKwargDataflowGraphData<GraphInputName, SlotName> const &data)
{
  return OpenKwargDataflowGraphView<GraphInputName, SlotName>::template create<
    ViewFromOpenKwargDataflowGraphView<GraphInputName, SlotName>>(data);
}

} // namespace FlexFlow

#endif
