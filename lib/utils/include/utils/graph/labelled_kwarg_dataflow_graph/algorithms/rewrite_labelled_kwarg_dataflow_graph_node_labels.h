#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_KWARG_DATAFLOW_GRAPH_ALGORITHMS_REWRITE_LABELLED_KWARG_DATAFLOW_GRAPH_NODE_LABELS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_KWARG_DATAFLOW_GRAPH_ALGORITHMS_REWRITE_LABELLED_KWARG_DATAFLOW_GRAPH_NODE_LABELS_H

#include "utils/graph/labelled_kwarg_dataflow_graph/labelled_kwarg_dataflow_graph.h"
#include "utils/graph/labelled_open_kwarg_dataflow_graph/algorithms/rewrite_labelled_open_kwarg_dataflow_graph_node_labels.h"
#include "utils/graph/labelled_kwarg_dataflow_graph/algorithms/view_as_labelled_open_kwarg_dataflow_graph.h"

namespace FlexFlow {

template <typename NodeLabel,
          typename ValueLabel,
          typename SlotName,
          typename F,
          typename NewNodeLabel = 
            std::invoke_result_t<F, Node const &, NodeLabel const &>>
LabelledKwargDataflowGraphView<NewNodeLabel, ValueLabel, SlotName>
  rewrite_labelled_kwarg_dataflow_graph_node_labels(LabelledKwargDataflowGraphView<NodeLabel, ValueLabel, SlotName> const &g, F f) 
{
  return rewrite_labelled_open_kwarg_dataflow_graph_node_labels<NodeLabel, ValueLabel, int, SlotName, F, NewNodeLabel>(
    view_as_labelled_open_kwarg_dataflow_graph<NodeLabel, ValueLabel, int, SlotName>(g), f);
}
  
} // namespace FlexFlow

#endif
