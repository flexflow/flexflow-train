#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_REWRITE_LABELLED_OPEN_KWARG_DATAFLOW_GRAPH_LABELS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_REWRITE_LABELLED_OPEN_KWARG_DATAFLOW_GRAPH_LABELS_H

#include "utils/containers/generate_map.h"
#include "utils/graph/labelled_open_kwarg_dataflow_graph/algorithms/open_kwarg_dataflow_graph_view_with_labelling.h"
#include "utils/graph/labelled_open_kwarg_dataflow_graph/labelled_open_kwarg_dataflow_graph.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/get_all_open_kwarg_dataflow_values.h"

namespace FlexFlow {

template <typename NodeLabel,
          typename ValueLabel,
          typename GraphInputName,
          typename SlotName,
          typename F,
          typename NewNodeLabel =
              std::invoke_result_t<F, Node const &, NodeLabel const &>,
          typename NewValueLabel = std::invoke_result_t<
              F,
              OpenKwargDataflowValue<GraphInputName, SlotName> const &,
              ValueLabel const &>>
LabelledOpenKwargDataflowGraphView<NewNodeLabel,
                                   NewValueLabel,
                                   GraphInputName,
                                   SlotName>
    rewrite_labelled_open_kwarg_dataflow_graph_labels(
        LabelledOpenKwargDataflowGraphView<NodeLabel,
                                           ValueLabel,
                                           GraphInputName,
                                           SlotName> const &g,
        F f) {
  auto get_new_node_label = [&](Node const &n) -> NewNodeLabel {
    return f(n, g.at(n));
  };

  auto get_new_value_label =
      [&](OpenKwargDataflowValue<GraphInputName, SlotName> const &v)
      -> NewValueLabel { return f(v, g.at(v)); };

  std::unordered_map<Node, NewNodeLabel> node_labels =
      generate_map(get_nodes(g), get_new_node_label);
  std::unordered_map<OpenKwargDataflowValue<GraphInputName, SlotName>,
                     NewValueLabel>
      value_labels = generate_map(get_all_open_kwarg_dataflow_values(g),
                                  get_new_value_label);
  return open_kwarg_dataflow_graph_view_with_labelling(
      g, node_labels, value_labels);
}

} // namespace FlexFlow

#endif
