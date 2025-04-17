#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_DATAFLOW_GRAPH_ALGORITHMS_REWRITE_VALUE_LABELS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_DATAFLOW_GRAPH_ALGORITHMS_REWRITE_VALUE_LABELS_H

#include "utils/graph/labelled_dataflow_graph/algorithms/view_as_labelled_open_dataflow_graph.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/rewrite_value_labels.h"

namespace FlexFlow {

template <
    typename NodeLabel,
    typename ValueLabel,
    typename F,
    typename NewValueLabel =
        std::invoke_result_t<F, OpenDataflowValue const &, ValueLabel const &>>
LabelledDataflowGraphView<NodeLabel, NewValueLabel> rewrite_value_labels(
    LabelledDataflowGraphView<NodeLabel, ValueLabel> const &g, F f) {
  return rewrite_value_labels<NodeLabel, ValueLabel, F, NewValueLabel>(
      view_as_labelled_open_dataflow_graph(g), f);
}

} // namespace FlexFlow

#endif
