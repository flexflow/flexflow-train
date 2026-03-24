#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_KWARG_DATAFLOW_GRAPH_ALGORITHMS_KWARG_DATAFLOW_GRAPH_VIEW_WITH_LABELLING_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_KWARG_DATAFLOW_GRAPH_ALGORITHMS_KWARG_DATAFLOW_GRAPH_VIEW_WITH_LABELLING_H

#include "utils/graph/kwarg_dataflow_graph/kwarg_dataflow_graph_view.h"
#include "utils/graph/labelled_kwarg_dataflow_graph/i_labelled_kwarg_dataflow_graph_view.h"
#include "utils/graph/labelled_kwarg_dataflow_graph/labelled_kwarg_dataflow_graph_view.h"

namespace FlexFlow {

template <typename NodeLabel, typename OutputLabel, typename SlotName>
struct KwargDataflowGraphLabellingWrapper final
    : public ILabelledKwargDataflowGraphView<NodeLabel, OutputLabel, SlotName> {
public:
  KwargDataflowGraphLabellingWrapper() = delete;
  KwargDataflowGraphLabellingWrapper(
      KwargDataflowGraphView<SlotName> const &unlabelled,
      std::unordered_map<Node, NodeLabel> const &node_labels,
      std::unordered_map<KwargDataflowOutput<SlotName>, OutputLabel> const
          &output_labels)
      : unlabelled(unlabelled), node_labels(node_labels),
        output_labels(output_labels) {}

  std::unordered_set<Node> query_nodes(NodeQuery const &q) const override {
    return this->unlabelled.query_nodes(q);
  }

  std::unordered_set<KwargDataflowEdge<SlotName>>
      query_edges(KwargDataflowEdgeQuery<SlotName> const &q) const override {
    return this->unlabelled.query_edges(q);
  }

  std::unordered_set<KwargDataflowOutput<SlotName>> query_outputs(
      KwargDataflowOutputQuery<SlotName> const &q) const override {
    return this->unlabelled.query_outputs(q);
  }

  NodeLabel at(Node const &n) const override {
    return this->node_labels.at(n);
  }

  OutputLabel at(KwargDataflowOutput<SlotName> const &v) const override {
    return this->output_labels.at(v);
  }

  KwargDataflowGraphLabellingWrapper *clone() const override {
    return new KwargDataflowGraphLabellingWrapper{
        this->unlabelled,
        this->node_labels,
        this->output_labels,
    };
  }

private:
  KwargDataflowGraphView<SlotName> unlabelled;
  std::unordered_map<Node, NodeLabel> node_labels;
  std::unordered_map<KwargDataflowOutput<SlotName>, OutputLabel> output_labels;
};

template <typename NodeLabel, typename OutputLabel, typename SlotName>
LabelledKwargDataflowGraphView<NodeLabel, OutputLabel, SlotName>
    kwarg_dataflow_graph_view_with_labelling(
        KwargDataflowGraphView<SlotName> const &g,
        std::unordered_map<Node, NodeLabel> const &node_labels,
        std::unordered_map<KwargDataflowOutput<SlotName>, OutputLabel> const
            &value_labels) {
  return LabelledKwargDataflowGraphView<NodeLabel, OutputLabel, SlotName>::
      template create<
          KwargDataflowGraphLabellingWrapper<NodeLabel, OutputLabel, SlotName>>(
          g, node_labels, value_labels);
}
} // namespace FlexFlow

#endif
