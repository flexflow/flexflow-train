#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_KWARG_DATAFLOW_GRAPH_ALGORITHMS_MATERIALIZE_LABELLED_KWARG_DATAFLOW_GRAPH_VIEW_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_KWARG_DATAFLOW_GRAPH_ALGORITHMS_MATERIALIZE_LABELLED_KWARG_DATAFLOW_GRAPH_VIEW_H

#include "utils/graph/instances/unordered_set_labelled_open_kwarg_dataflow_graph.h"
#include "utils/graph/labelled_kwarg_dataflow_graph/labelled_kwarg_dataflow_graph.h"
namespace FlexFlow {

template <typename NodeLabel, typename ValueLabel, typename SlotName>
LabelledKwargDataflowGraph<NodeLabel, ValueLabel, SlotName>
    materialize_labelled_kwarg_dataflow_graph_view(
        LabelledKwargDataflowGraphView<NodeLabel, ValueLabel, SlotName> const
            &view) {
  return LabelledKwargDataflowGraph<NodeLabel, ValueLabel, SlotName>::
      template create_copy_of<
          UnorderedSetLabelledOpenKwargDataflowGraph<NodeLabel,
                                                     ValueLabel,
                                                     std::monostate,
                                                     SlotName>>(view);
}

} // namespace FlexFlow

#endif
