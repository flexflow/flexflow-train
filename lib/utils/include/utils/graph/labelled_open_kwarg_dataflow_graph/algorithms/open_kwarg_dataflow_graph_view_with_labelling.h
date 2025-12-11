#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_OPEN_KWARG_DATAFLOW_GRAPH_VIEW_WITH_LABELLING_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_OPEN_KWARG_DATAFLOW_GRAPH_VIEW_WITH_LABELLING_H

#include "utils/graph/labelled_open_kwarg_dataflow_graph/i_labelled_open_kwarg_dataflow_graph_view.h"
#include "utils/graph/labelled_open_kwarg_dataflow_graph/labelled_open_kwarg_dataflow_graph_view.h"
#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_graph_view.h"

namespace FlexFlow {

template <typename NodeLabel, typename ValueLabel, typename GraphInputName, typename SlotName>
struct OpenKwargDataflowGraphLabellingWrapper final
    : public ILabelledOpenKwargDataflowGraphView<NodeLabel, ValueLabel, GraphInputName, SlotName> {
public:
  OpenKwargDataflowGraphLabellingWrapper() = delete;
  OpenKwargDataflowGraphLabellingWrapper(
      OpenKwargDataflowGraphView<GraphInputName, SlotName> const &unlabelled,
      std::unordered_map<Node, NodeLabel> const &node_labels,
      std::unordered_map<OpenKwargDataflowValue<GraphInputName, SlotName>, ValueLabel> const &value_labels)
      : unlabelled(unlabelled), node_labels(node_labels),
        value_labels(value_labels) {}

  std::unordered_set<Node> query_nodes(NodeQuery const &q) const override {
    return this->unlabelled.query_nodes(q);
  }

  std::unordered_set<OpenKwargDataflowEdge<GraphInputName, SlotName>>
      query_edges(OpenKwargDataflowEdgeQuery<GraphInputName, SlotName> const &q) const override {
    return this->unlabelled.query_edges(q);
  }

  std::unordered_set<KwargDataflowOutput<SlotName>>
      query_outputs(KwargDataflowOutputQuery<SlotName> const &q) const override {
    return this->unlabelled.query_outputs(q);
  }

  std::unordered_set<KwargDataflowGraphInput<GraphInputName>> get_inputs() const override {
    return this->unlabelled.get_inputs();
  }

  NodeLabel at(Node const &n) const override {
    return this->node_labels.at(n);
  }

  ValueLabel at(OpenKwargDataflowValue<GraphInputName, SlotName> const &v) const override {
    return this->value_labels.at(v);
  }

  OpenKwargDataflowGraphLabellingWrapper *clone() const override {
    return new OpenKwargDataflowGraphLabellingWrapper{
        this->unlabelled,
        this->node_labels,
        this->value_labels,
    };
  }

private:
  OpenKwargDataflowGraphView<GraphInputName, SlotName> unlabelled;
  std::unordered_map<Node, NodeLabel> node_labels;
  std::unordered_map<OpenKwargDataflowValue<GraphInputName, SlotName>, ValueLabel> value_labels;
};

template <typename NodeLabel,
          typename ValueLabel,
          typename GraphInputName, 
          typename SlotName>
LabelledOpenKwargDataflowGraphView<NodeLabel, ValueLabel, GraphInputName, SlotName> 
  open_kwarg_dataflow_graph_view_with_labelling(
    OpenKwargDataflowGraphView<GraphInputName, SlotName> const &g, 
    std::unordered_map<Node, NodeLabel> const &node_labels,
    std::unordered_map<OpenKwargDataflowValue<GraphInputName, SlotName>, ValueLabel> const &value_labels) 
{
  return LabelledOpenKwargDataflowGraphView<NodeLabel, ValueLabel, GraphInputName, SlotName>::template create<
    OpenKwargDataflowGraphLabellingWrapper<NodeLabel, ValueLabel, GraphInputName, SlotName>>(
      g, node_labels, value_labels);
}


} // namespace FlexFlow

#endif
