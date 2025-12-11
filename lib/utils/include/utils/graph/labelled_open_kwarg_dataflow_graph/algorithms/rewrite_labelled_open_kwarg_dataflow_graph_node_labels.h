#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_REWRITE_LABELLED_OPEN_KWARG_DATAFLOW_GRAPH_NODE_LABELS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_REWRITE_LABELLED_OPEN_KWARG_DATAFLOW_GRAPH_NODE_LABELS_H

#include "utils/graph/labelled_open_kwarg_dataflow_graph/algorithms/rewrite_labelled_open_kwarg_dataflow_graph_labels.h"
#include "utils/overload.h"

namespace FlexFlow {

template <
    typename NodeLabel,
    typename ValueLabel,
    typename GraphInputName,
    typename SlotName,
    typename F,
    typename NewNodeLabel =
        std::invoke_result_t<F, Node const &, NodeLabel const &>>
LabelledOpenKwargDataflowGraphView<NewNodeLabel, ValueLabel, GraphInputName, SlotName> rewrite_labelled_open_kwarg_dataflow_graph_node_labels(
    LabelledOpenKwargDataflowGraphView<NodeLabel, ValueLabel, GraphInputName, SlotName> const &g, F f) {

  return rewrite_labelled_open_kwarg_dataflow_graph_labels(
    g,
    overload {
      [&](Node const &n, NodeLabel const &l) { return f(n, l); },
      [](OpenKwargDataflowValue<GraphInputName, SlotName> const &, ValueLabel const &l) { return l; },
    });
}

} // namespace FlexFlow

#endif
