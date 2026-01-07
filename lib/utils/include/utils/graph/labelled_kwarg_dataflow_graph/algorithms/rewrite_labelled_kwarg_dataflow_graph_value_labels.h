#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_KWARG_DATAFLOW_GRAPH_ALGORITHMS_REWRITE_LABELLED_KWARG_DATAFLOW_GRAPH_VALUE_LABELS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_KWARG_DATAFLOW_GRAPH_ALGORITHMS_REWRITE_LABELLED_KWARG_DATAFLOW_GRAPH_VALUE_LABELS_H

#include "utils/graph/labelled_kwarg_dataflow_graph/algorithms/view_as_labelled_open_kwarg_dataflow_graph.h"
#include "utils/graph/labelled_kwarg_dataflow_graph/labelled_kwarg_dataflow_graph.h"
#include "utils/graph/labelled_open_kwarg_dataflow_graph/algorithms/rewrite_labelled_open_kwarg_dataflow_graph_value_labels.h"

namespace FlexFlow {

template <typename NodeLabel,
          typename ValueLabel,
          typename SlotName,
          typename F,
          typename NewValueLabel =
              std::invoke_result_t<F,
                                   KwargDataflowOutput<SlotName> const &,
                                   ValueLabel const &>>
LabelledKwargDataflowGraphView<NodeLabel, NewValueLabel, SlotName>
    rewrite_labelled_kwarg_dataflow_graph_value_labels(
        LabelledKwargDataflowGraphView<NodeLabel, ValueLabel, SlotName> const
            &g,
        F f) {
  auto label_func = [&](OpenKwargDataflowValue<int, SlotName> const &v,
                        ValueLabel const &l) -> NewValueLabel {
    return v.template visit<NewValueLabel>(overload{
        [](KwargDataflowGraphInput<int> const &) -> NewValueLabel { PANIC(); },
        [&](KwargDataflowOutput<SlotName> const &o) -> NewValueLabel {
          return f(o, l);
        }});
  };

  return rewrite_labelled_open_kwarg_dataflow_graph_value_labels(
      view_as_labelled_open_kwarg_dataflow_graph<NodeLabel,
                                                 ValueLabel,
                                                 int,
                                                 SlotName>(g),
      label_func);
}

} // namespace FlexFlow

#endif
