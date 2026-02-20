#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_REWRITE_LABELLED_OPEN_KWARG_DATAFLOW_GRAPH_VALUE_LABELS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_REWRITE_LABELLED_OPEN_KWARG_DATAFLOW_GRAPH_VALUE_LABELS_H

#include "utils/graph/labelled_open_kwarg_dataflow_graph/algorithms/rewrite_labelled_open_kwarg_dataflow_graph_labels.h"
#include "utils/overload.h"

namespace FlexFlow {

template <typename NodeLabel,
          typename ValueLabel,
          typename GraphInputName,
          typename SlotName,
          typename F,
          typename NewValueLabel = std::invoke_result_t<
              F,
              OpenKwargDataflowValue<GraphInputName, SlotName> const &,
              ValueLabel const &>>
LabelledOpenKwargDataflowGraphView<NodeLabel,
                                   NewValueLabel,
                                   GraphInputName,
                                   SlotName>
    rewrite_labelled_open_kwarg_dataflow_graph_value_labels(
        LabelledOpenKwargDataflowGraphView<NodeLabel,
                                           ValueLabel,
                                           GraphInputName,
                                           SlotName> const &g,
        F f) {

  return rewrite_labelled_open_kwarg_dataflow_graph_labels(
      g,
      overload{
          [](Node const &, NodeLabel const &l) { return l; },
          [&](OpenKwargDataflowValue<GraphInputName, SlotName> const &v,
              ValueLabel const &l) { return f(v, l); },
      });
}

} // namespace FlexFlow

#endif
