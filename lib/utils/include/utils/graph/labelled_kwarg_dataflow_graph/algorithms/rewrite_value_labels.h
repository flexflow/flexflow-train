#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_KWARG_DATAFLOW_GRAPH_ALGORITHMS_REWRITE_VALUE_LABELS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_KWARG_DATAFLOW_GRAPH_ALGORITHMS_REWRITE_VALUE_LABELS_H

#include "utils/graph/labelled_kwarg_dataflow_graph/labelled_kwarg_dataflow_graph.h"
#include "utils/graph/labelled_open_kwarg_dataflow_graph/algorithms/rewrite_value_labels.h"

namespace FlexFlow {

template <typename NodeLabel,
          typename ValueLabel,
          typename SlotName,
          typename F,
          typename NewValueLabel = 
            std::invoke_result_t<F, KwargDataflowOutput<SlotName> const &, ValueLabel const &>>
LabelledKwargDataflowGraph<NodeLabel, NewValueLabel, SlotName>
  rewrite_value_labels(LabelledKwargDataflowGraph<NodeLabel, ValueLabel, SlotName> const &g, F f) 
{
  return rewrite_value_labels<NodeLabel, ValueLabel, int, SlotName, F, NewValueLabel>(
    view_as_labelled_open_kwarg_dataflow_graph(g), f);
}

} // namespace FlexFlow

#endif
